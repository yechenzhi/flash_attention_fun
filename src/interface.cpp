#include <torch/extension.h>
#include "utils.cuh"

torch::Tensor forward_v1(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    auto O = torch::empty_like(Q);
    launch_flash_attn_v1(Q, K, V, O);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_v1", &forward_v1, "Flash Attention V1 (Naive)");
}