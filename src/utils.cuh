#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <type_traits>

// 宏定义
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 可以在这里定义一些公用的 Math 函数，比如 fast_exp, fast_rsqrt 等
#define WARP_SIZE 32
#define SHFL_ENTIRE_WARP_MASK 0xffffffff
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define COPY_SIZE_BF16 8

#define LDMATRIX_MAT_SIZE 8
#define ROWS_PER_FRAGMENT LDMATRIX_MAT_SIZE
#define COLS_PER_FRAGMENT LDMATRIX_MAT_SIZE

#define MMA_M_FRAGMENTS_PER_ITER 2 // (MMA_M / LDMATRIX_MAT_SIZE)
#define MMA_N_FRAGMENTS_PER_ITER 1 // (MMA_N / LDMATRIX_MAT_SIZE)
#define MMA_K_FRAGMENTS_PER_ITER 2 // (MMA_K / LDMATRIX_MAT_SIZE)

// async copy from global memory to shared memory (16 Bytes = 8 x bf16)
DEVICE void cp_async_ca(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
        : : "r"(smem), "l"(glob_ptr));
}

DEVICE void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

DEVICE void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

DEVICE void cp_async_wait_prev() {
    asm volatile("cp.async.wait_group 1;\n");
}

// ldmatrix load from shared memory to registers
DEVICE void ldmatrix_m8n8x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void* smem_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem));
}

DEVICE void ldmatrix_m8n8x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, void* smem_ptr) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem));
}

// MMA operation: D = A * B + C 
// m16n8k16 A: 4 regs (16x16), B: 2 regs (16x8), C/D: 4 regs (16x8)
// A/B are in bf16, C/D are in f32
DEVICE void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    const uint32_t& a0, const uint32_t& a1, const uint32_t& a2, const uint32_t& a3,
    const uint32_t& b0, const uint32_t& b1,
    const float& c0, const float& c1, const float& c2, const float& c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), 
          "r"(b0), "r"(b1), 
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

template <int M_fragments, int N_fragments, int K_fragments>
DEVICE void warp_fragment_mma_f32_accum(
    uint32_t (&regs_A)[M_fragments][K_fragments],
    uint32_t (&regs_B)[N_fragments][K_fragments],
    float (&regs_C)[M_fragments][N_fragments * 2]) {
    #pragma unroll
    for (int k = 0; k < K_fragments; k += MMA_K_FRAGMENTS_PER_ITER) {
        #pragma unroll
        for (int m = 0; m < M_fragments; m += MMA_M_FRAGMENTS_PER_ITER) {
            #pragma unroll
            for (int n = 0; n < N_fragments; n += MMA_N_FRAGMENTS_PER_ITER) {
                mma_m16n8k16_bf16(
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1],
                    regs_A[m][k],
                    regs_A[m + 1][k],
                    regs_A[m][k + 1],
                    regs_A[m + 1][k + 1],
                    regs_B[n][k],
                    regs_B[n][k + 1],
                    regs_C[m][n * 2],
                    regs_C[m][n * 2 + 1],
                    regs_C[m + 1][n * 2],
                    regs_C[m + 1][n * 2 + 1]);
            }
        }
    }
}

template <typename value_t, int M_fragments, int N_fragments>
DEVICE void
convert_to_16_bit_dtype(float (&src_float)[M_fragments][N_fragments * 2],
                        uint32_t (&dest_uint)[M_fragments][N_fragments]) {
    using value2_t =
        std::conditional_t<std::is_same_v<value_t, half>, half2, nv_bfloat162>;

    float2(&src)[M_fragments][N_fragments] =
        reinterpret_cast<float2(&)[M_fragments][N_fragments]>(src_float);
    value2_t(&dest)[M_fragments][N_fragments] =
        reinterpret_cast<value2_t(&)[M_fragments][N_fragments]>(dest_uint);
    #pragma unroll
    for (int m = 0; m < M_fragments; ++m) {
        #pragma unroll
        for (int n = 0; n < N_fragments; ++n) {
            if constexpr (std::is_same_v<value_t, half>) {
                dest[m][n] = __float22half2_rn(src[m][n]);
            } else {
                dest[m][n] = __float22bfloat162_rn(src[m][n]);
            }
        }
    }
}
