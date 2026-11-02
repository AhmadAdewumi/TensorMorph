
#include "../include/tensormorph/cuda/memory.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensormorph::cuda {
namespace {
void check_cuda_error(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}
} // namespace

DeviceBuffer::DeviceBuffer(std::size_t bytes) : size_bytes_(bytes) {
  if (bytes == 0) {
    return;
  }

  check_cuda_error(cudaMalloc(&pointer_, bytes), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() {
  if (pointer_ != nullptr) {
    cudaFree(pointer_);
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }

    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
  }

  return *this;
}

void *DeviceBuffer::data() noexcept { return pointer_; }

const void *DeviceBuffer::data() const noexcept { return pointer_; }

std::size_t DeviceBuffer::size_bytes() const noexcept { return size_bytes_; }

void copy_host_to_device(void *destination, const void *source,
                         std::size_t bytes) {
  if (bytes == 0) {
    return;
  }

  check_cuda_error(
      cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy H2D");
}

void copy_device_to_host(void *destination, const void *source,
                         std::size_t bytes) {
  if (bytes == 0) {
    return;
  }

  check_cuda_error(
      cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
      "cudaMemcpy D2H");
}

void copy_device_to_device(void *destination, const void *source,
                           std::size_t bytes) {
  if (bytes == 0) {
    return;
  }

  check_cuda_error(
      cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice),
      "cudaMemcpy D2D");
}

} // namespace tensormorph::cuda
