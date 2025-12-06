#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_flash_attn_v1(
    const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V, torch::Tensor& O
);

torch::Tensor forward_v1(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    auto O = torch::empty_like(Q);
    launch_flash_attn_v1(Q, K, V, O);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v1", &forward_v1, "Flash Attention V1 (Naive)");
}