import os

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
import torch
import triton
import triton.language as tl


@triton.jit
def mxfp8_dot_scaled_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sA_ptr,
    sB_ptr,
    stride_am,
    stride_ak,  # A_codes: [M, K]
    stride_bk,
    stride_bn,  # B_codes: [K, N]
    stride_cm,
    stride_cn,  # C: [M, N]
    stride_meta_a_m: tl.constexpr,
    stride_meta_a_g: tl.constexpr,
    stride_meta_b_n: tl.constexpr,
    stride_meta_b_g: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    tl.static_assert(BLOCK_K % GROUP_K == 0)
    BLOCK_K_S: tl.constexpr = BLOCK_K // GROUP_K

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ACC = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k_scales = tl.arange(0, BLOCK_K_S)

    for k0 in tl.range(0, K, BLOCK_K):
        kv = k0 + tl.arange(0, BLOCK_K)

        a_mask = ((rm[:, None] < M) & (kv[None, :] < K)).to(tl.int1)
        b_mask = ((kv[:, None] < K) & (rn[None, :] < N)).to(tl.int1)

        A_idx = rm[:, None] * stride_am + kv[None, :] * stride_ak
        B_idx = kv[:, None] * stride_bk + rn[None, :] * stride_bn

        A_codes = tl.load(A_ptr + A_idx, mask=a_mask, other=0)
        B_codes = tl.load(B_ptr + B_idx, mask=b_mask, other=0)

        k_group_index_base = k0 // GROUP_K
        sA_blk = tl.load(sA_ptr + rm[:, None] * stride_meta_a_m +
                         (k_group_index_base + offs_k_scales[None, :]) * stride_meta_a_g)
        sB_blk = tl.load(sB_ptr + rn[:, None] * stride_meta_b_n +
                         (k_group_index_base + offs_k_scales[None, :]) * stride_meta_b_g)

        ACC = tl.dot_scaled(A_codes, sA_blk, "e4m3", B_codes, sB_blk, "e4m3", ACC)

    C_idx = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + C_idx, ACC, mask=c_mask)


def run_mxfp8_dot_scaled_kernel(a_codes: torch.Tensor, sA_grouped: torch.Tensor, b_codes: torch.Tensor,
                                sB_grouped: torch.Tensor, M: int, N: int, K: int, block_m: int = 32, block_n: int = 16,
                                block_k: int = 64, group_k: int = 32, num_warps: int = 4,
                                num_stages: int = 2) -> torch.Tensor:
    device = a_codes.device
    c = torch.empty((M, N), dtype=torch.float32, device=device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    mxfp8_dot_scaled_kernel[grid](a_codes, b_codes, c, M, N, K, sA_grouped, sB_grouped, a_codes.stride(0),
                                  a_codes.stride(1), b_codes.stride(0), b_codes.stride(1), c.stride(0), c.stride(1),
                                  sA_grouped.stride(0), sA_grouped.stride(1), sB_grouped.stride(0),
                                  sB_grouped.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                                  GROUP_K=group_k, num_warps=num_warps, num_stages=num_stages)
    return c


def decode_fp8_e4m3(codes: torch.Tensor) -> torch.Tensor:
    x = codes.to(torch.int16)
    s = (x >> 7) & 0x1
    e = (x >> 3) & 0xF
    m = x & 0x7
    bias = 7
    is_zero = (e == 0) & (m == 0)
    is_sub = (e == 0) & (m != 0)
    is_norm = (e > 0) & (e < 0xF)

    out = torch.zeros_like(x, dtype=torch.float32)
    out = torch.where(is_sub, (m.float() / 8.0) * (2.0**(1 - bias)), out)
    out = torch.where(is_norm, (1.0 + m.float() / 8.0) * (2.0**(e.float() - bias)), out)
    out = torch.where(is_zero, out, torch.where(s.bool(), -out, out))
    return out


@torch.no_grad()
def mxfp8_scaled_reference_grouped(a_codes, b_codes, sA_grouped, sB_grouped, M: int, N: int, K: int, GROUP_K: int):
    a_fp = decode_fp8_e4m3(a_codes)
    b_fp = decode_fp8_e4m3(b_codes)

    sA = (2**(sA_grouped.float() - 127.0)).repeat_interleave(GROUP_K, dim=1)[:, :K]
    sB = (2**(sB_grouped.float() - 127.0)).repeat_interleave(GROUP_K, dim=1)[:, :K]

    a_scaled = a_fp * sA
    b_scaled = b_fp * sB.T
    return a_scaled @ b_scaled


def test_mxfp8_matmul(M: int, N: int, K: int, num_warps: int = 4):
    device = "cuda"

    # FP8 E4M3 encoding for 1.0: 0x38
    fp8_one = 0x38

    a_codes = torch.full((M, K), fp8_one, dtype=torch.uint8, device=device)
    b_codes = torch.full((K, N), fp8_one, dtype=torch.uint8, device=device)

    GROUP_K = 32
    KG = K // GROUP_K
    deltaA = 0
    deltaB = 0
    sA = torch.randint(127 - deltaA, 128 + deltaA, (M, KG), dtype=torch.uint8, device="cuda")
    sB = torch.randint(127 - deltaB, 128 + deltaB, (N, KG), dtype=torch.uint8, device="cuda")

    c = run_mxfp8_dot_scaled_kernel(a_codes, sA, b_codes, sB, M, N, K, group_k=GROUP_K, num_warps=num_warps)
    c_ref = mxfp8_scaled_reference_grouped(a_codes=a_codes, b_codes=b_codes, sA_grouped=sA, sB_grouped=sB, M=M, N=N,
                                           K=K, GROUP_K=GROUP_K)

    torch.testing.assert_close(c, c_ref, rtol=1e-3, atol=0.0)
    return c, c_ref


if __name__ == "__main__":
    for nw in [1, 2, 4, 8]:
        for bm in [16, 32, 64, 128, 256]:
            for bn in [8, 16, 32, 64, 128, 256]:
                for bk in [64, 128, 256]:
                    M, N, K = bm, bn, bk
                    print(f"[FP8] M={M} N={N} K={K} | BLOCK_M={bm} BLOCK_N={bn} BLOCK_K={bk} GROUP_K=32 NUM_WARPS={nw}")
                    c, c_ref = test_mxfp8_matmul(M, N, K, num_warps=nw)
                    print("OK")
