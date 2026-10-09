#include <torch/extension.h>

// 所有算子的 schema 集中在这里声明，Python 侧通过 torch.ops.cudalab.<name> 调用。
// CUDA 实现位于 csrc/kernels/ 下，各自用 TORCH_LIBRARY_IMPL(cudalab, CUDA, m) 注册。
TORCH_LIBRARY(cudalab, m) {
  m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

// 使 _C.so 成为可 import 的 Python 模块；import 时上面的静态注册随之执行。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
