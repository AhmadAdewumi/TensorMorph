
#include "tensormorph/cuda/error.hpp"
#include <cstddef>
#include <cuda_runtime_api.h>
#include <curand_mtgp32_kernel.h>
#include <driver_types.h>
namespace tensormorph::cuda::kernels {

namespace {
constexpr unsigned int tile = 32;

__global__ void transpose_kernel(const float *input, float *output,
                                 std::size_t rows, std::size_t columns) {
  __shared__ float tile_data[tile][tile + 1];

  const std::size_t x =
      static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.x;

  const std::size_t y =
      static_cast<std::size_t>(blockIdx.y) * tile + threadIdx.y;

  // load elements
  if (x < columns && y < rows) {
    tile_data[threadIdx.y][threadIdx.x] = input[y * columns + x];
  }

  __syncthreads();

  // switch x and y position and boom transposition, WOWW!!!, WOWW!!!,
  // INTERESTING!!!
  const std::size_t transposed_x =
      static_cast<std::size_t>(blockIdx.y) * tile + threadIdx.x;

  const std::size_t transposed_y =
      static_cast<std::size_t>(blockIdx.x) * tile + threadIdx.y;

  if (transposed_x < rows && transposed_y < columns) {
    output[transposed_y * rows + transposed_x] =
        tile_data[threadIdx.x][threadIdx.y];
  }
}

void transpose(const float *input, float *output, std::size_t rows,
               std::size_t columns, int device_index, void *stream) {
  if (rows == 0 || columns == 0) {
    return;
  }

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  const dim3 blockSize(tile, tile);

  const dim3 gridSize(static_cast<unsigned int>((columns + tile - 1) / tile),
                      static_cast<unsigned int>((rows + tile - 1) / tile));

  //-- 0 -> no dynamic shared memory
  transpose_kernel<<<gridSize, blockSize, 0,
                     static_cast<cudaStream_t>(stream)>>>(input, output, rows,
                                                          columns);

  TM_CUDA_CHECK(cudaGetLastError());
}
} // namespace
} // namespace tensormorph::cuda::kernels
