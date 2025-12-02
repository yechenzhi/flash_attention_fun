#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

// 宏定义
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 可以在这里定义一些公用的 Math 函数，比如 fast_exp, fast_rsqrt 等

// ------------------------------------------------------------
// Launcher 函数声明 (让 interface.cpp 知道它们的存在)
// ------------------------------------------------------------

void launch_flash_attn_v1(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V, torch::Tensor& O
);
