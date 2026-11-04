
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace tensormorph::cuda {

inline void check(cudaError_t error, const char *expression, const char *file,
                  int line) {
  if (error == cudaSuccess) {
    return;
  }

  throw std::runtime_error(std::string("CUDA error: ") +
                           cudaGetErrorString(error) +
                           " | expression: " + expression + " | file: " + file +
                           " | line: " + std::to_string(line));
}

} // namespace tensormorph::cuda

#define TM_CUDA_CHECK(expression)                                              \
  ::tensormorph::cuda::check((expression), #expression, __FILE__, __LINE__)
