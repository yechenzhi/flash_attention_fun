#include "../utils.cuh"
#include <cuda_runtime.h>

// V1 Kernel 实现 (Dummy Copy 用于测试)
template<typename T>
__global__ void flash_attn_kernel_v1(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int N, int d 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 简单的 Copy 逻辑：O = Q
    // 确保不越界
    if (idx < N * d) {
        O[idx] = Q[idx]; 
    }
}

// V1 Launcher
void launch_flash_attn_v1(
    const torch::Tensor& Q, 
    const torch::Tensor& K, 
    const torch::Tensor& V, 
    torch::Tensor& O
) {
    // 简单设置 grid/block
    int total_elements = Q.numel();
    int head_dim = Q.size(3); // 假设 shape 是 [B, S, H, D]
    
    dim3 block(128);
    dim3 grid((total_elements + 127) / 128);

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
            flash_attn_kernel_v1<scalar_t><<<grid, block>>>(
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                O.data_ptr<scalar_t>(),
                total_elements / head_dim, 
                head_dim
            );
        })
    );
    
    // 检查 kernel 启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}