
#include "tensormorph/cuda/error.hpp"
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace tensormorph::cuda::kernels {
namespace {
__global__ void reduce_kernel(const float *input, float *output,
                              std::size_t noOfElements) {

  extern __shared__ float shared[];
  const unsigned int tid = threadIdx.x;

  // each thread will be responsible for loading 2 values
  //  calc the input index
  // say we have blockIdx.x = 0, blockDim.x = 4, index for thread 1 --> 0*4*2+1
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2 + threadIdx.x;

  float value = 0.0F;

  // first loaded values
  if (index < noOfElements) {
    value += input[index];
  }

  // second loaded values
  if ((index + blockDim.x) < noOfElements) {
    value += input[index + blockDim.x];
  }

  // each thread writes its local sum into the shared memory
  shared[tid] = value;
  // we wait till all threads finish writing
  __syncthreads();

  // reduction loop, after thread local summations
  //-- stride >>= 1 --> strides = strides/2
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // after the reduction, shared[0] should contain sum of all input elements in
  // that block,  only thread 0 gets to execute this
  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

void reduce_sum(const float *input, float *output, std::size_t noOfElements,
                int device_index, void *stream) {
  TM_CUDA_CHECK(cudaSetDevice(device_index));

  // empty arr case
  if (noOfElements == 0) {
    TM_CUDA_CHECK(cudaMemsetAsync(output, 0, sizeof(float),
                                  static_cast<cudaStream_t>(stream)));
    return;
  }

  constexpr std::size_t blockSize = 256;
  const unsigned int gridSize =
      (noOfElements + (blockSize * 2) - 1) / (blockSize * 2);

  /*
   * NB: we can have many blocks undergo reduction
   */
  // an array to each block sum floats on GPU
  float *partial_block_sum = nullptr;

  // alloc mem on the GPU
  TM_CUDA_CHECK(cudaMalloc(&partial_block_sum, gridSize * sizeof(float)));

  reduce_kernel<<<static_cast<unsigned int>(gridSize), blockSize,
                  blockSize * sizeof(float),
                  static_cast<cudaStream_t>(stream)>>>(input, output,
                                                       noOfElements);
  TM_CUDA_CHECK(cudaGetLastError());

  /*
   * if only one block was launched, out result is already reduced to 1 value,
   * we just copy result to the output
   */
  if (gridSize == 1) {
    TM_CUDA_CHECK(cudaMemcpyAsync(output, partial_block_sum, sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  static_cast<cudaStream_t>(stream)));
  } else {
    // we call reduce kernel again
    reduce_kernel<<<1, blockSize, blockSize * sizeof(float),
                    static_cast<cudaStream_t>(stream)>>>(partial_block_sum,
                                                         output, gridSize);
    TM_CUDA_CHECK(cudaGetLastError());
  }

  TM_CUDA_CHECK(
      cudaFreeAsync(partial_block_sum, static_cast<cudaStream_t>(stream)));
}
} // namespace

} // namespace tensormorph::cuda::kernels
