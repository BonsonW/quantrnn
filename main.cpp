#include <torch/extension.h>

torch::Tensor forward(torch::Tensor _A, torch::Tensor _B) ;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
