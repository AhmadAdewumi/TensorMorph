#include "tensormorph/cuda/error.hpp"
#include "tensormorph/cuda/kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace tensormorph::cuda::kernels {

namespace {

constexpr unsigned int threads = 256;

__global__ void add_kernel(const float *a, const float *b, float *output,
                           std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < count) {
    output[i] = a[i] + b[i];
  }
}

__global__ void multiply_kernel(const float *a, const float *b, float *output,
                                std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < count) {
    output[i] = a[i] * b[i];
  }
}

__global__ void scale_kernel(const float *input, float scalar, float *output,
                             std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < count) {
    output[i] = input[i] * scalar;
  }
}

__global__ void relu_kernel(const float *input, float *output,
                            std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < count) {
    output[i] = fmaxf(input[i], 0.0F);
  }
}

unsigned int grid_size(std::size_t count) {
  const std::size_t blocks = (count + threads - 1) / threads;

  if (blocks > static_cast<std::size_t>(2147483647U)) {
    throw std::overflow_error("CUDA grid dimension too large");
  }

  return static_cast<unsigned int>(blocks);
}

} // namespace

void add(const float *a, const float *b, float *output, std::size_t count,
         int device_index, void *stream) {
  if (count == 0)
    return;

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  add_kernel<<<grid_size(count), threads, 0,
               static_cast<cudaStream_t>(stream)>>>(a, b, output, count);

  TM_CUDA_CHECK(cudaGetLastError());
}

void multiply(const float *a, const float *b, float *output, std::size_t count,
              int device_index, void *stream) {
  if (count == 0)
    return;

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  multiply_kernel<<<grid_size(count), threads, 0,
                    static_cast<cudaStream_t>(stream)>>>(a, b, output, count);

  TM_CUDA_CHECK(cudaGetLastError());
}

void scale(const float *input, float scalar, float *output, std::size_t count,
           int device_index, void *stream) {
  if (count == 0)
    return;

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  scale_kernel<<<grid_size(count), threads, 0,
                 static_cast<cudaStream_t>(stream)>>>(input, scalar, output,
                                                      count);

  TM_CUDA_CHECK(cudaGetLastError());
}

void relu(const float *input, float *output, std::size_t count,
          int device_index, void *stream) {
  if (count == 0)
    return;

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  relu_kernel<<<grid_size(count), threads, 0,
                static_cast<cudaStream_t>(stream)>>>(input, output, count);

  TM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensormorph::cuda::kernels
