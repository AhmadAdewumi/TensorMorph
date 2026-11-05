
#include "tensormorph/cuda/error.hpp"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector_types.h>
namespace tensormorph::cuda::kernels {
namespace {

__global__ void matmul_kernel(const float *a, const float *b, float *output,
                              std::size_t m, std::size_t k, std::size_t n) {

  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t col =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0F;

  for (std::size_t p = 0; p < k; ++p) {
    sum += a[row * n + p] * b[p * n + col];
  }

  output[row * n + col] = sum;
}

// HOST SIDE
void matmul(const float *a, const float *b, float *output, std::size_t m,
            std::size_t k, std::size_t n, int device_index, void *stream) {
  if (m == 0 || n == 0)
    return;

  TM_CUDA_CHECK(cudaSetDevice(device_index));
  constexpr unsigned int tile = 16;

  const dim3 blockSize(tile, tile);

  //ceiling
  const dim3 gridSize(static_cast<unsigned int>(n + tile - 1) / tile,
                      static_cast<unsigned int>(n + tile - 1) / tile);
  matmul_kernel<<<gridSize, blockSize, 0, static_cast<cudaStream_t>(stream)>>>(
      a, b, output, m, k, n);

  TM_CUDA_CHECK(cudaGetLastError());
}

} // namespace
} // namespace tensormorph::cuda::kernels
