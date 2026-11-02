
#include <cstddef>
#include <stdexcept>
namespace tensormorph::cuda::kernels {
__global__ void vector_add_kernel(const float *a, const float *b, float *output,
                                  const std::size_t noOfElements) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < noOfElements) {
    output[idx] = a[idx] + b[idx];
  }
}

void vector_add(const float *a, const float *b, float *output,
                std::size_t noOfElements) {
  if (noOfElements == 0) {
    return;
  }

  constexpr unsigned int threads = 256;

  const unsigned int blocks =
      static_cast<unsigned int>((noOfElements + threads - 1) / threads);

  vector_add_kernel<<<blocks, threads>>>(a, b, output, noOfElements);

  cudaError_t error = cudaGetLastError();

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
} // namespace tensormorph::cuda::kernels
