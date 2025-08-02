#!/usr/bin/env python3
# fp8_scaled_bench_fixed.py
#
#   python fp8_scaled_bench_fixed.py --m 4096 --n 4096 --k 8192 --rep 100
#
import argparse, torch, triton, triton.language as tl

# ─ compile-time tile sizes ───────────────────────────────────────────
BLOCK_M = 16
BLOCK_N = 8
BLOCK_K = 32          # m16 n8 k32 → dot_scaled 기본 판넬

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
                              offs_k[None, :]*stride_ak)        # 16×32
        B = tl.load(B_ptr + offs_k[:, None]*stride_bk +
                              offs_n[None, :]*stride_bn)        # 32×8  (잘못)
        _sA = tl.load(sA_ptr + offs_m)[:,None]
        _sB = tl.load(sB_ptr + offs_n)[None, :]
        acc = tl.dot_scaled(A, _sA, "e4m3",
                            B, _sB, "e4m3",
                            acc)
    tl.store(C_ptr + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn, acc)

def bench(M, N, K, atype="e4m3fn", btype="e4m3fn"):
    assert (M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0), \
        f"dims must be multiples of {BLOCK_M}×{BLOCK_N}×{BLOCK_K}"

    dtype = torch.float8_e4m3fn
    A = torch.ones(M, K, device="cuda", dtype=dtype)
    B = torch.ones(N, K, device="cuda", dtype=dtype).t()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    if False:
        sA = torch.full((M,), 130, dtype=torch.uint8, device="cuda")
        sB = torch.full((N,), 127, dtype=torch.uint8, device="cuda")
    else:
                                # DEBUG: Check which warp each element comes from
        # Set different values for different warps to see the pattern
        # sA = torch.tensor([100, 101, 102, 103,   # Warp 0 should read these
                        #   110, 111, 112, 113,   # Warp 1 should read these
                        #   120, 121, 122, 123,   # Warp 2 should read these
                        #   130, 131, 132, 133], dtype=torch.uint8, device="cuda")  # Warp 3 should read these
        # TEST aScale: bScale=127(1.0), aScale varies per row
        sB = 127 + torch.arange(N, dtype=torch.uint8, device="cuda").flip(0)
        sA = 126 + torch.arange(M, dtype=torch.uint8, device="cuda")   # [126,127,128,129,...] = [0.5,1,2,4,...]
        print("Expected: rows should vary if aScale works, all same if aScale broken")
        # sB = torch.full((N,), 127, dtype=torch.uint8, device="cuda")

    grid = (M // BLOCK_M, N // BLOCK_N)
    print("\n=== Running main kernel ===")
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

    # DEBUG: Analyze the pattern
    print("\n=== DEBUG ANALYSIS ===")
    print("Expected pattern (each row should use different aScale):")
    for i in range(min(16, M)):
        expected_scale = sA[i].item()
        expected_fp8_val = 2.0 ** (expected_scale - 127.0)
        expected_result = int(1.0 * expected_fp8_val * 1.0 * K)  # A=1, B=1, K=32
        print(f"Row {i:2d}: sA[{i}]={expected_scale} → scale={expected_fp8_val:.6f} → result={expected_result}")

    print("\nActual results (Fix):")
    fix_results = C[:16, 0].to(torch.int32)
    for i in range(16):
        print(f"Row {i:2d}: {fix_results[i].item()}")

    print("\nPattern analysis:")
    # Check if results repeat every 4 rows
    for i in range(4):
        row_values = [fix_results[j].item() for j in range(i, 16, 4)]
        print(f"Every 4th row starting from {i}: {row_values}")
        if len(set(row_values)) == 1:
            print(f"  → All same: {row_values[0]} (PROBLEM!)")
        else:
            print(f"  → Different values (GOOD!)")
    # print("=== ttir ===")
    print(compiled.asm["ptx"])

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    m = 16
    n = 8
    k = 32
    args = pa.parse_args()
    bench(m, n, k)