
#include "../include/tensormorph/cuda/memory.hpp"
#include "tensormorph/cuda/error.hpp"
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
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

void select_device(int device_index) {
  check_cuda_error(cudaSetDevice(device_index), "cudaSetDevice");
}
} // namespace

DeviceBuffer::DeviceBuffer(std::size_t bytes, int device_index)
    : size_bytes_(bytes), device_index_(device_index) {
  if (device_index < 0) {
    throw std::invalid_argument("CUDA device index cannot be negative");
  }

  if (bytes == 0) {
    return;
  }

  select_device(device_index);

  check_cuda_error(cudaMalloc(&pointer_, bytes), "cudaMalloc");
}

DeviceBuffer::~DeviceBuffer() noexcept {
  reset();
  // if (pointer_ != nullptr) {
  //   cudaFree(pointer_);
  // }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : pointer_(std::exchange(other.pointer_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)),
      device_index_(std::exchange(other.device_index_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    reset();

    pointer_ = std::exchange(other.pointer_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    device_index_ = std::exchange(other.device_index_, 0);
  }

  return *this;
}

void *DeviceBuffer::data() noexcept { return pointer_; }

const void *DeviceBuffer::data() const noexcept { return pointer_; }

std::size_t DeviceBuffer::size_bytes() const noexcept { return size_bytes_; }

DeviceBuffer::operator bool() const noexcept { return pointer_ != nullptr; }

void DeviceBuffer::reset() noexcept {
  if (pointer_) {
    cudaSetDevice(device_index_);
    cudaFree(pointer_);
  }

  pointer_ = nullptr;
  size_bytes_ = 0;
  device_index_ = 0;
}

void validate_copy(void *destination, const void *source, std::size_t bytes,
                   const char *operation) {
  if (bytes == 0) {
    return;
  }

  if (destination == nullptr || source == nullptr) {
    throw std::invalid_argument(std::string(operation) +
                                ": received a null pointer");
  }
}

void copy_host_to_device(void *destination, const void *source,
                         std::size_t bytes, int device_index) {
  validate_copy(destination, source, bytes, "H2D copy");

  // select_device(device_index);
  TM_CUDA_CHECK(cudaSetDevice(device_index));
  TM_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
}

void copy_device_to_host(void *destination, const void *source,
                         std::size_t bytes, int device_index) {
  validate_copy(destination, source, bytes, "D2H copy");

  // select_device(device_index);
  TM_CUDA_CHECK(cudaSetDevice(device_index));
  TM_CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost));
}

void copy_device_to_device(void *destination, const void *source,
                           std::size_t bytes, int device_index) {
  validate_copy(destination, source, bytes, "D2D copy");

  TM_CUDA_CHECK(cudaSetDevice(device_index));
  TM_CUDA_CHECK(
      cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice));
}

} // namespace tensormorph::cuda
