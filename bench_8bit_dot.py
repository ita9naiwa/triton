import os
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

import torch
import triton
import triton.language as tl
import time

# Set global scratch allocator for TMA descriptors
def cuda_allocator(size, align, stream):
    return torch.cuda.caching_allocator_alloc(size, stream=stream)

triton.set_allocator(cuda_allocator)


@triton.jit
def matmul_8bit_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Use tl.range for pipelining & TMA
    for k in tl.range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        # A load: [BLOCK_M, BLOCK_K] - 8bit
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=a_mask, other=0).to(tl.float32)

        # B load: [BLOCK_K, BLOCK_N] - 8bit
        b_mask = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=b_mask, other=0).to(tl.float32)

        # Regular dot (not dot_scaled)
        acc += tl.dot(a, b)

    # Store result
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc, mask=c_mask)


def benchmark_8bit_matmul(M, N, K, block_m, block_n, block_k, num_warps=2, num_stages=4):
    device = "cuda"
    # 8bit uint8 tensors with padding to create strided accesses that favor TMA
    pad_k = 0
    pad_n = 128
    a_big = torch.randint(0, 255, (M, K + pad_k), dtype=torch.uint8, device=device)
    b_big = torch.randint(0, 255, (K, N + pad_n), dtype=torch.uint8, device=device)
    a = a_big[:, :K]
    b = b_big[:, :N]
    c = torch.empty((M, N), dtype=torch.float32, device=device)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    print(f"\n{'='*80}")
    print(f"Benchmarking 8bit matmul: M={M}, N={N}, K={K}")
    print(f"Block sizes: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}, NUM_WARPS={num_warps}")
    print(f"{'='*80}")

    kernel = matmul_8bit_kernel[grid]

    # Compile
    ret = kernel(
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Check for TMA
    has_tma = 'tensormap' in ret.asm.get('ptx', '')
    print(f"TMA enabled: {'✅ YES' if has_tma else '❌ NO'}")

    # Warmup
    for _ in range(50):
        kernel(a, b, c, M, N, K, a.stride(0), a.stride(1),
               b.stride(0), b.stride(1), c.stride(0), c.stride(1),
               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
               num_warps=num_warps, num_stages=num_stages)
    torch.cuda.synchronize()

    # Benchmark
    num_iters = 200
    start = time.time()
    for _ in range(num_iters):
        kernel(a, b, c, M, N, K, a.stride(0), a.stride(1),
               b.stride(0), b.stride(1), c.stride(0), c.stride(1),
               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
               num_warps=num_warps, num_stages=num_stages)
    torch.cuda.synchronize()
    end = time.time()

    elapsed_ms = (end - start) * 1000 / num_iters
    flops = 2 * M * N * K
    tflops = flops / (elapsed_ms * 1e9)

    print(f"Time per iteration: {elapsed_ms:.3f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS")
    print(f"{'='*80}\n")

    return tflops, has_tma


if __name__ == "__main__":
    configs = [
        # (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)
        (1024, 1024, 1024, 64, 64, 256, 4),
        (2048, 2048, 2048, 64, 64, 256, 4),
        (4096, 4096, 4096, 64, 64, 256, 4),
        (1024, 1024, 1024, 64, 64, 256, 4),
        (2048, 2048, 2048, 64, 64, 256, 4),
    ]

    print("\n" + "="*80)
    print("8-bit Regular Dot Benchmark (with TMA pass)")
    print("="*80)

    results = []
    for M, N, K, bm, bn, bk, nw in configs:
        tflops, has_tma = benchmark_8bit_matmul(M, N, K, bm, bn, bk, nw)
        results.append((M, N, K, bm, bn, bk, tflops, has_tma))

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'Size':<15} {'Block':<20} {'TFLOPS':<10} {'TMA':<5}")
    print("-" * 80)
    for M, N, K, bm, bn, bk, tflops, has_tma in results:
        size_str = f"{M}x{N}x{K}"
        block_str = f"{bm}x{bn}x{bk}"
        tma_str = "YES" if has_tma else "NO"
        print(f"{size_str:<15} {block_str:<20} {tflops:<10.2f} {tma_str:<5}")
    print("="*80)

