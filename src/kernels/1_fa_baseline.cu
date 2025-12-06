#include "../utils.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16 = nv_bfloat16;

template<typename T, int Br, int Bc, int num_warps, int D>
__global__ void flash_attn_kernel_v1(
    const T* __restrict__ Q_in,
    const T* __restrict__ K_in,
    const T* __restrict__ V_in,
    T* __restrict__ O_in,
    int batch_stride, int seq_stride, int head_stride,
    int batch_size, int seq_len, int num_heads, int dim, 
    int n_Q_blocks, int n_KV_blocks
) {
    // PROLOGUE
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int sample_id = blockIdx.z;
    int head_id = blockIdx.y;
    int Q_block_id = blockIdx.x;

    // global memory offsets
    const int sample_head_offset = sample_id * batch_stride + head_id * head_stride; 
    const int QO_gmem_block_offset = sample_head_offset + Q_block_id * Br * seq_stride;
    const int KV_gmem_block_offset_base = sample_head_offset;

    Q_in += QO_gmem_block_offset;
    K_in += KV_gmem_block_offset_base;
    V_in += KV_gmem_block_offset_base;
    O_in += QO_gmem_block_offset;

    const bf16* Q = reinterpret_cast<const bf16*>(Q_in);
    const bf16* K = reinterpret_cast<const bf16*>(K_in);
    const bf16* V = reinterpret_cast<const bf16*>(V_in);
    bf16* O = reinterpret_cast<bf16*>(O_in);

    // shared memory pointers
    extern __shared__ __align__(128) bf16 smem[];
    bf16* s_Q = smem;
    bf16* s_K = s_Q + Br * D;
    bf16* s_V = s_K + Bc * D;
    bf16* s_O = s_Q; // resue s_Q for output

    // load Q block to shared memory
    // Q,K,V,O all the same layouts in global memory <-> shared memory
    const int s_row = tid / (D / COPY_SIZE_BF16);
    const int s_col = tid % (D / COPY_SIZE_BF16);
    const int s_stride = WARP_SIZE * num_warps / (D / COPY_SIZE_BF16);

    for(int row_offset = 0; row_offset < Br; row_offset += s_stride){
        const bf16* glob_ptr = &Q[(row_offset + s_row) * seq_stride + s_col * COPY_SIZE_BF16];
        bf16* smem_ptr = &s_Q[(row_offset + s_row) * D + s_col * COPY_SIZE_BF16];
        cp_async_ca(smem_ptr, glob_ptr);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // load s_Q to registers reg_Q (Freagment A)
    // s_Q: 64x128. 4 warps.
    // each warp load 64 / 4 x 128 = 16 x 128
    // mma_m16n8k16_bf16 
    // 16x128 is [2x16] x [8x8]. 8x8 is a tile/fragment. 16x16 is composed of 2x2 times 8x8 fragments

    constexpr int row_fragments_per_iter = 2;
    constexpr int col_fragments_per_iter = 2; // 2x2 [8x8] fragments per iteration for ldmatrix_m8n8x4
    constexpr int rows_per_iter = ROWS_PER_FRAGMENT * row_fragments_per_iter;
    constexpr int cols_per_iter = COLS_PER_FRAGMENT * row_fragments_per_iter;

    const int QO_rows_per_warp = Br / num_warps; //16
    const int KV_rows_per_warp = Bc / num_warps; //16
    const int QO_fragments_per_warp = QO_rows_per_warp / ROWS_PER_FRAGMENT; //2
    // const int dim_fragments = dim / COLS_PER_FRAGMENT; // 16
    constexpr int dim_fragments = D / COLS_PER_FRAGMENT;
    uint32_t reg_Q[QO_fragments_per_warp][dim_fragments]; 

    const int thread_row = lane_id % rows_per_iter;
    const int thread_col_fragment = lane_id /  rows_per_iter;

    // load s_Q to reg_Q
    for(int r = 0; r < QO_fragments_per_warp; r += row_fragments_per_iter){
        const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
        for(int c = 0; c < dim_fragments; c += col_fragments_per_iter){
            const int cur_col = (thread_col_fragment + c) * COLS_PER_FRAGMENT;
            ldmatrix_m8n8x4(reg_Q[r][c], reg_Q[r+1][c], reg_Q[r][c+1], reg_Q[r+1][c+1]
                , &s_Q[(warp_id * QO_rows_per_warp + cur_row) * D + cur_col]);
        }
    }

    // initialize accumulators
    const int KV_fragments_per_warp = Bc / ROWS_PER_FRAGMENT; // 64 / 8 = 8

    uint32_t reg_K[KV_fragments_per_warp][dim_fragments];
    uint32_t reg_V[dim_fragments][KV_fragments_per_warp];
    float reg_S_accum[QO_fragments_per_warp][KV_fragments_per_warp * 2];
    uint32_t reg_P[QO_fragments_per_warp][KV_fragments_per_warp];
    float reg_O_accum[QO_fragments_per_warp][dim_fragments * 2];
    uint32_t reg_O[QO_fragments_per_warp][dim_fragments]; 
    
    for(int r = 0; r < QO_fragments_per_warp; r++){
        for(int c = 0; c < dim_fragments * 2; c++){
            reg_O_accum[r][c] = 0.0f;
        }
    }

    const float softmax_scale = rsqrt(static_cast<float>(D));
    float m[QO_fragments_per_warp];
    float l[QO_fragments_per_warp];

    constexpr float neg_inf = -cuda::std::numeric_limits<float>::infinity();

    for(int q = 0; q < QO_fragments_per_warp; q++){
        m[q] = neg_inf;
        l[q] = 0.0f;
    }

    // MAIN LOOP over KV blocks
    n_KV_blocks = CEIL_DIV(seq_len, Bc);
    for(int kv_block_id = 0; kv_block_id < n_KV_blocks; kv_block_id++){
        //load K from global memory to shared memory
        for(int row_offset = 0; row_offset < Bc; row_offset += s_stride){
            const bf16* glob_ptr = &K[(row_offset + s_row) * seq_stride + s_col * COPY_SIZE_BF16];
            bf16* smem_ptr = &s_K[(row_offset + s_row) * D + s_col * COPY_SIZE_BF16];
            cp_async_ca(smem_ptr, glob_ptr);
        }
        K += Bc * seq_stride;
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        for(int r = 0; r < QO_fragments_per_warp; r++){
            for(int c = 0; c < KV_fragments_per_warp * 2; c++){
                reg_S_accum[r][c] = 0.0f;
            }
        }

        //load K to registers reg_K
        for(int r = 0; r < KV_fragments_per_warp; r += row_fragments_per_iter){
            const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
            for(int c = 0; c < dim_fragments; c += col_fragments_per_iter){
                const int cur_col = (thread_col_fragment + c) * COLS_PER_FRAGMENT;
                ldmatrix_m8n8x4(reg_K[r][c], reg_K[r+1][c], reg_K[r][c+1], reg_K[r+1][c+1]
                    , &s_K[cur_row * D + cur_col]);
            }
        }

        // MMA to compute S_accm
        warp_fragment_mma_f32_accum<QO_fragments_per_warp, KV_fragments_per_warp, dim_fragments>(
            reg_Q,
            reg_K,
            reg_S_accum
        );

        //scale S_accm
        for(int q = 0; q < QO_fragments_per_warp; q++){
            for(int k = 0; k < KV_fragments_per_warp * 2; k++){
                reg_S_accum[q][k] = reg_S_accum[q][k] * softmax_scale;
            }
        }

        //compute softmax
        //calc_row_max
        float m_next[QO_fragments_per_warp];
        for (int q = 0; q < QO_fragments_per_warp; ++q) {
            m_next[q] = m[q];

            // Calculate max for row across all in-thread registers.
            for (int k = 0; k < KV_fragments_per_warp * 2; ++k) {
                m_next[q] = max(m_next[q], reg_S_accum[q][k]);
            }

            // Group reduction
            m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 2),
                            m_next[q]);
            m_next[q] = max(__shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, m_next[q], 1),
                            m_next[q]);
        }
        // in fact, m_next is m_curr here, m is m_prev

        //scale l and o
        for (int q = 0; q < QO_fragments_per_warp; ++q) {
            float scale = expf(m[q] - m_next[q]);
            m[q] = m_next[q];
            l[q] *= scale;
            for (int d = 0; d < dim_fragments * 2; ++d) {
                reg_O_accum[q][d] *= scale;
            }
        }
        //exp reg_S_accum and update l
        for(int q = 0; q < QO_fragments_per_warp; q++){
            for(int k = 0; k < KV_fragments_per_warp * 2; k++){
                reg_S_accum[q][k] = expf(reg_S_accum[q][k] - m[q]);
                l[q] += reg_S_accum[q][k];
            }
        }

        //convert reg_S_accum to reg_P
        convert_to_16_bit_dtype<bf16,QO_fragments_per_warp,KV_fragments_per_warp>(reg_S_accum, reg_P);
        
        //load V from global memory to shared memory
        for(int row_offset = 0; row_offset < Bc; row_offset += s_stride){
            const bf16* glob_ptr = &V[(row_offset + s_row) * seq_stride + s_col * COPY_SIZE_BF16];
            bf16* smem_ptr = &s_V[(row_offset + s_row) * D + s_col * COPY_SIZE_BF16];
            cp_async_ca(smem_ptr, glob_ptr);
        }
        V += Bc * seq_stride;
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // load V to registers reg_V
        for(int c = 0; c < dim_fragments; c += col_fragments_per_iter){
            const int cur_col = (thread_col_fragment + c) * COLS_PER_FRAGMENT;
            for(int r = 0; r < KV_fragments_per_warp; r += row_fragments_per_iter){
                const int cur_row = thread_row + r * ROWS_PER_FRAGMENT;
                ldmatrix_m8n8x4_trans(reg_V[c][r], reg_V[c][r+1], reg_V[c+1][r], reg_V[c+1][r+1]
                    , &s_V[cur_row * D + cur_col]);
            }
        }

        //MMA to compute O_accm
        warp_fragment_mma_f32_accum<QO_fragments_per_warp, dim_fragments, KV_fragments_per_warp>(
            reg_P,
            reg_V,
            reg_O_accum
        );
    }

    // EPILOGUE
    // Final warp-level reduction for l
    // we always use 4 threads for one row
    for (int q = 0; q < QO_fragments_per_warp; ++q) {
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 2);
        l[q] += __shfl_xor_sync(SHFL_ENTIRE_WARP_MASK, l[q], 1);
    }
    // Final row-wise O softmax normalization.
    for (int q = 0; q < QO_fragments_per_warp; ++q) {
        for (int d = 0; d < dim_fragments * 2; ++d) {
            reg_O_accum[q][d] /= l[q];
        }
    }

    //convert reg_O_accum to reg_O
    convert_to_16_bit_dtype<bf16,QO_fragments_per_warp,dim_fragments>(reg_O_accum, reg_O);

    //store reg_O to shared memory s_Q
    const int thread_row_O = lane_id / 4;
    const int thread_col_O = (lane_id % 4) * 2;
    for(int r = 0; r < QO_fragments_per_warp; r += 1){
        const int cur_row = thread_row_O + r * ROWS_PER_FRAGMENT;
        for(int c = 0; c < dim_fragments; c += 1){
            const int cur_col = thread_col_O + c * COLS_PER_FRAGMENT;
            reinterpret_cast<uint32_t *>(
                &s_Q[(warp_id * QO_rows_per_warp + cur_row) * D + cur_col])[0] = reg_O[r][c];
        }
    }

    __syncthreads();

    //store s_O to global memory
    for(int row_offset = 0; row_offset < Br; row_offset += s_stride){
        bf16* glob_ptr = &O[(row_offset + s_row) * seq_stride + s_col * COPY_SIZE_BF16];
        bf16* smem_ptr = &s_Q[(row_offset + s_row) * D + s_col * COPY_SIZE_BF16];
        reinterpret_cast<uint4 *>(glob_ptr)[0] = reinterpret_cast<uint4 *>(smem_ptr)[0];
    }
}

// V1 Launcher
void launch_flash_attn_v1(
    const torch::Tensor& Q, 
    const torch::Tensor& K, 
    const torch::Tensor& V, 
    torch::Tensor& O
) {
    // Q shape: [B, S, H, D]
    // for L2 cache hit rate, use seq_len as bIdx.x, num_heads as bIdx.y, batch_size as bIdx.z
    const int num_warps = 4;
    const int num_threads = num_warps * WARP_SIZE;
    const int Br = 64;
    const int Bc = 64;

    int batch_size = Q.size(0);
    int seq_len = Q.size(1);
    int num_heads = Q.size(2);
    int dim = Q.size(3); 

    int batch_stride = Q.stride(0);
    int seq_stride = Q.stride(1);
    int head_stride = Q.stride(2);

    const int n_Q_blocks = CEIL_DIV(seq_len, Br);
    const int n_KV_blocks = CEIL_DIV(seq_len, Bc);

    dim3 blockDim(128);
    dim3 gridDim{n_Q_blocks, num_heads, batch_size};
    const int smem_bytes = (Br + Bc + Bc) * dim * 2 + 1024; // Q + K + V

    // 【修改点】：支持 BF16 和 FP16
    // 参数1: 支持类型1 (FP16)
    // 参数2: 支持类型2 (BF16)
    // 参数3: 输入 Tensor 的类型
    // 参数4: Kernel 名称
    // 参数5: Lambda 函数
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, 
        at::ScalarType::BFloat16, 
        Q.scalar_type(), 
        "flash_attn_kernel_v1", 
        ([&] {
            auto kernel_func = flash_attn_kernel_v1<scalar_t, Br, Bc, num_warps, 128>;

            cudaFuncSetAttribute(
                    kernel_func,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 
                    smem_bytes
                );
            
            kernel_func<<<gridDim, blockDim, smem_bytes>>>(
                    Q.data_ptr<scalar_t>(),
                    K.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    O.data_ptr<scalar_t>(),
                    batch_stride, seq_stride, head_stride,
                    batch_size, seq_len, num_heads, dim,
                    n_Q_blocks, n_KV_blocks
                );
        })
    );
    
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}