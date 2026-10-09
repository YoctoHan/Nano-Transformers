#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace cudalab {

template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace cudalab
