#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "common.cuh"

namespace cudalab {

template <typename scalar_t>
__global__ void vector_add_kernel(const scalar_t* __restrict__ a,
                                  const scalar_t* __restrict__ b,
                                  scalar_t* __restrict__ out,
                                  int64_t n) {
  // 用 int64 计算全局下标：元素数超过 2^31 时 int 会溢出
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}

at::Tensor vector_add_cuda(const at::Tensor& a, const at::Tensor& b) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape, got ", a.sizes(),
              " and ", b.sizes());
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");

  // 多卡时保证 kernel 发射到输入所在的 GPU
  const c10::cuda::CUDAGuard device_guard(a.device());

  at::Tensor out = at::empty_like(a);
  const int64_t n = a.numel();
  if (n == 0) {
    return out;
  }

  constexpr int kThreadsPerBlock = 256;
  const int64_t num_blocks = ceil_div<int64_t>(n, kThreadsPerBlock);
  // 必须使用 PyTorch 的当前 stream，才能与前后的 torch 算子保持正确的执行顺序
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, a.scalar_type(), "vector_add_cuda", [&] {
    vector_add_kernel<scalar_t><<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        a.const_data_ptr<scalar_t>(), b.const_data_ptr<scalar_t>(),
        out.mutable_data_ptr<scalar_t>(), n);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}

TORCH_LIBRARY_IMPL(cudalab, CUDA, m) {
  m.impl("vector_add", &vector_add_cuda);
}

}  // namespace cudalab
