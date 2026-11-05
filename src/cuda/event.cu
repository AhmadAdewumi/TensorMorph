#include "tensormorph/cuda/error.hpp"
#include "tensormorph/cuda/event.hpp"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
#include <utility>

namespace tensormorph::cuda {

Event::Event(int device_index) : device_index_(device_index) {
  if (device_index < 0) {
    throw std::invalid_argument("Negative CUDA device index");
  }

  TM_CUDA_CHECK(cudaSetDevice(device_index_));

  cudaEvent_t event{};
  TM_CUDA_CHECK(cudaEventCreate(&event));

  handle_ = event;
}

Event::~Event() noexcept {
  if (handle_) {
    cudaSetDevice(device_index_);
    cudaEventDestroy(static_cast<cudaEvent_t>(handle_));
  }
}

Event::Event(Event &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      device_index_(std::exchange(other.device_index_, 0)) {}

Event &Event::operator=(Event &&other) noexcept {
  if (this != &other) {
    if (handle_) {
      if (handle_) {
        cudaSetDevice(device_index_);
        cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
      }
    }

    handle_ = std::exchange(other.handle_, nullptr);
    device_index_ = std::exchange(other.device_index_, 0);
  }

  return *this;
}

void Event::record(void *stream) {
  TM_CUDA_CHECK(cudaSetDevice(device_index_));

  TM_CUDA_CHECK(
      cudaEventRecord(static_cast<cudaEvent_t>(handle_), cudaStream_t(stream)));
}

void Event::synchronize() const {
  TM_CUDA_CHECK(cudaSetDevice(device_index_));

  TM_CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(handle_)));
}

float Event::elapsed_ms(const Event &end) const {
  if (device_index_ != end.device_index_) {
    throw std::invalid_argument("Events belong to different devices");
  }

  float milliseconds = 0.0F;

  TM_CUDA_CHECK(cudaSetDevice(device_index_));
  TM_CUDA_CHECK(cudaEventElapsedTime(&milliseconds,
                                     static_cast<cudaEvent_t>(handle_),
                                     static_cast<cudaEvent_t>(end.handle_)));

  return milliseconds;
}

void *Event::native_handle() const noexcept { return handle_; }

} // namespace tensormorph::cuda
