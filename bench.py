#!/usr/bin/env python3
# fp8_scaled_bench_fixed.py
#
#   python fp8_scaled_bench_fixed.py --m 4096 --n 4096 --k 8192 --rep 100
#
import argparse, torch, triton, triton.language as tl

# ─ compile-time tile sizes ───────────────────────────────────────────
BLOCK_M = 16
BLOCK_N = 8
BLOCK_K = 32

def scaleDot_ref(A, B, sA, sB):
    def fp8e8m0_to_float32(scale):
        return 2.0 ** (scale - 127.0)

    tA = A.to(torch.float32) * fp8e8m0_to_float32(sA[:, None])
    tB = B.to(torch.float32) * fp8e8m0_to_float32(sB[None, :])
    return torch.matmul(tA, tB)

@triton.jit
def dot_kernel(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               sA_ptr, sB_ptr,
               stride_am, stride_ak,
               stride_bk, stride_bn,
               stride_cm, stride_cn,
               BLOCK_M: tl.constexpr,
               BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr):

    pid_m = tl.program_id(0)                       # CTA id in M
    pid_n = tl.program_id(1)                       # CTA id in N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # [16]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # [8]

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)             # [32]

        A = tl.load(A_ptr + offs_m[:, None]*stride_am +
                              offs_k[None, :]*stride_ak)
        B = tl.load(B_ptr + offs_k[:, None]*stride_bk +
                              offs_n[None, :]*stride_bn)
        # acc = tl.dot(A, B, acc)
        _sA = tl.load(sA_ptr + offs_m)[:,None]
        _sB = tl.load(sB_ptr + offs_n)[:, None]
        acc = tl.dot_scaled(A, _sA, "e4m3",
                            B, _sB, "e4m3",
                            acc)
    tl.store(C_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn, acc)

def bench(M, N, K, atype="e4m3fn", btype="e4m3fn"):
    assert (M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0), \
        f"dims must be multiples of {BLOCK_M}×{BLOCK_N}×{BLOCK_K}"

    dtype = torch.float8_e4m3fn
    A = torch.ones(M, K, device="cuda", dtype=dtype)
    B = torch.ones(K, N, device="cuda", dtype=dtype)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    sB = (127 + torch.arange(N, dtype=torch.uint8, device="cuda")).flip(0)
    sA = 126 + torch.arange(M, dtype=torch.uint8, device="cuda")

    grid = (M // BLOCK_M, N // BLOCK_N)
    compiled = dot_kernel[grid](
        A, B, C, M, N, K,
        sA, sB,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    C1 = scaleDot_ref(A, B, sA, sB)
    print("Fix: ", C[:16, :4].to(torch.int32))
    print("Ref: ", C1[:16, :4].to(torch.int32))
    # print(compiled.asm["ptx"])

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    m = 16
    n = 8
    k = 32
    print("M, N, K: ", m, n, k)
    args = pa.parse_args()
    bench(m, n, k)