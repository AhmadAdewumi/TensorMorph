#include "tensormorph/cuda/error.hpp"
#include "tensormorph/cuda/stream.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <new>
#include <stdexcept>
#include <utility>

namespace tensormorph::cuda {

Stream::Stream(int device_index) : device_index_(device_index) {
  if (device_index < 0) {
    throw std::invalid_argument("Negative CUDA device index");
  }

  TM_CUDA_CHECK(cudaSetDevice(device_index));

  cudaStream_t stream{};
  TM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  handle_ = stream;
}

Stream::~Stream() noexcept {
  if (handle_) {
    cudaSetDevice(device_index_);
    cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
  }
}

Stream::Stream(Stream &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      device_index_(std::exchange(other.device_index_, 0)) {}

Stream &Stream::operator=(Stream &&other) noexcept {
  if (this != &other) {
    if (handle_) {
      TM_CUDA_CHECK(cudaSetDevice(device_index_));
      TM_CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(handle_)));
    }

    handle_ = std::exchange(other.handle_, nullptr);
    device_index_ = std::exchange(other.device_index_, 0);
  }

  return *this;
}

// native cuda handle
void *Stream::native_handle() const noexcept { return handle_; }

int Stream::device_index() const noexcept { return device_index_; }

void Stream::synchronize() const noexcept {
  TM_CUDA_CHECK(cudaSetDevice(device_index_));
  TM_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(handle_)));
}
} // namespace tensormorph::cuda
