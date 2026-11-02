
#include "../include/tensormorph/cuda/runtime.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace tensormorph::cuda {
namespace {
void check_cuda_error(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}
} // namespace

int device_count() {
  int count = 0;

  check_cuda_error(cudaGetDeviceCount(&count), "cudaGetDeviceCount");

  return count;
}

DeviceInfo device_info(int device_id) {
  cudaDeviceProp properties{};

  check_cuda_error(cudaGetDeviceProperties(&properties, device_id),
                   "cudaGetDeviceProperties");

  DeviceInfo info{};
  info.id = device_id;

  std::size_t i = 0;
  for (; i < sizeof(info.name) - 1 && properties.name[i] != '\0'; ++i) {
    info.name[i] = properties.name[i];
  }

  info.name[i] = '\0';

  info.compute_major = properties.major;
  info.compute_minor = properties.minor;

  info.multiprocessor_count = properties.multiProcessorCount;
  info.warp_size = properties.warpSize;

  info.total_global_memory = properties.totalGlobalMem;
  info.shared_memory_per_block = properties.sharedMemPerBlock;
  info.registers_per_block = properties.regsPerBlock;

  info.async_engine_count = properties.asyncEngineCount;
  info.concurrent_kernels = properties.concurrentKernels;

  return info;
}

void set_device(int device_id) {
  check_cuda_error(cudaSetDevice(device_id), "cudaSetDevice");
}

void synchronize() {
  check_cuda_error(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}
} // namespace tensormorph::cuda
