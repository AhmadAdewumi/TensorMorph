#include "tensormorph/tensor.hpp"
#include "tensormorph/cuda/tensor.hpp"

#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace tensormorph {

Tensor::Tensor(std::vector<std::size_t> shape, Device target)
    : shape_(std::move(shape)), numel_(checked_numel(shape_)),
      storage_([&]() -> Storage {
        if (target.is_cpu()) {
          return CpuStorage{std::vector<float>(numel_, 0.0F)};
        }

#ifdef TENSORMORPH_HAS_CUDA
        if (target.is_gpu()) {
          return CudaStorage{
              cuda::DeviceBuffer{numel_ * sizeof(float), target.index}};
        }
#else
        if (target.is_gpu()) {
          throw std::runtime_error("Tensormorph built without cuda support");
        }
#endif
        throw std::invalid_argument("Unsupported device type");
      }()) {
  if (target.index < 0) {
    throw std::invalid_argument("Device index cannot be negative");
  }
}

Tensor::Tensor(std::vector<std::size_t> shape, Storage storage)
    : shape_(shape), numel_(checked_numel(shape)),
      storage_(std::move(storage)) {}

Tensor::Tensor(const Tensor &other)
    : shape_(other.shape_), numel_(other.numel_), storage_([&]() -> Storage {
        if (const auto *cpu = std::get_if<CpuStorage>(&other.storage_)) {
          return CpuStorage{cpu->values};
        }

        const auto &gpu = std::get<CudaStorage>(other.storage_);

        CudaStorage copy{cuda::DeviceBuffer{gpu.buffer.size_bytes(),
                                            gpu.buffer.device_index()}};

        cuda::copy_device_to_device(copy.buffer.data(), gpu.buffer.data(),
                                    gpu.buffer.size_bytes(),
                                    gpu.buffer.device_index());
        return copy;
      }()) {}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this == &other) {
    return *this;
  }

  Tensor copy(other);

  shape_ = std::move(copy.shape_);
  numel_ = copy.numel_;
  storage_ = std::move(copy.storage_);

  return *this;
}

Tensor Tensor::from_data(std::vector<std::size_t> shape,
                         std::vector<float> values, Device target) {
  const std::size_t expected = checked_numel(shape);
  if (values.size() != expected) {
    throw std::invalid_argument(
        "Tensor data size doesn not match the provided shape");
  }

  if (target.is_cpu()) {
    return Tensor(std::move(shape), Device::cpu());
  }

  if (target.is_gpu()) {
    return cuda_tensor_from_data(std::move(shape), std::move(values),
                                 target.index);
  }

  throw std::invalid_argument("Unsupported device type");
}

std::span<const std::size_t> Tensor::shape() const noexcept { return shape_; }

std::size_t Tensor::rank() const noexcept { return shape_.size(); }

std::size_t Tensor::numel() const noexcept { return numel_; }

Device Tensor::device() const noexcept {
  if (std::holds_alternative<CpuStorage>(storage_)) {
    return Device::cpu();
  }

  const auto &gpu = std::get<CudaStorage>(storage_);

  return Device::gpu(gpu.buffer.device_index());
}

std::span<float> Tensor::data() {
  auto *cpu = std::get_if<CpuStorage>(&storage_);

  if (cpu == nullptr) {
    throw std::runtime_error(
        "Tensor::data() cannot expose CUDA memory as a host span");
  }

  return cpu->values;
}

std::span<const float> Tensor::data() const {
  const auto *cpu = std::get_if<CpuStorage>(&storage_);

  if (cpu == nullptr) {
    throw std::runtime_error(
        "Tensor::data() cannot expose CUDA memory as a host span");
  }

  return cpu->values;
}

void *Tensor::device_data() {
  auto *gpu = std::get_if<CudaStorage>(&storage_);

  if (gpu == nullptr) {
    throw std::runtime_error("Tensor does not contain CUDA storage");
  }

  return gpu->buffer.data();
}

const void *Tensor::device_data() const {
  const auto *gpu = std::get_if<CudaStorage>(&storage_);

  if (gpu == nullptr) {
    throw std::runtime_error("Tensor does not contain CUDA storage");
  }

  return gpu->buffer.data();
}

Tensor Tensor::to(Device target) const {
  if (target == device()) {
    return *this;
  }

  if (target.is_cpu()) {
    if (const auto *cpu = std::get_if<CpuStorage>(&storage_)) {
      return Tensor(shape_, CpuStorage{cpu->values});
    }

#ifdef TENSORMORPH_HAS_CUDA
    const auto &gpu = std::get<CudaStorage>(storage_);

    std::vector<float> values(numel_);

    if (numel_ != 0) {
      cuda::copy_device_to_host(values.data(), gpu.buffer.data(),
                                numel_ * sizeof(float),
                                gpu.buffer.device_index());
    }

    return Tensor(shape_, CpuStorage{std::move(values)});
#else
    throw std::runtime_error("TensorMorph was built without CUDA support");
#endif
  }

  if (target.is_gpu()) {
    if (const auto *cpu = std::get_if<CpuStorage>(&storage_)) {
      return from_data(shape_, cpu->values, target);
    }

    const auto &source = std::get<CudaStorage>(storage_);

    CudaStorage destination{
        cuda::DeviceBuffer(numel_ * sizeof(float), target.index)};

    if (numel_ != 0) {
      // Cross-device copies are deliberately not implemented yet.
      if (source.buffer.device_index() != target.index) {
        throw std::runtime_error(
            "Cross-GPU tensor transfers are not implemented");
      }

      cuda::copy_device_to_device(destination.buffer.data(),
                                  source.buffer.data(), numel_ * sizeof(float),
                                  target.index);
    }

    return Tensor(shape_, std::move(destination));
  }

  throw std::invalid_argument("Unsupported target device");
}

std::size_t Tensor::checked_numel(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;

  for (const std::size_t dimension : shape) {
    if (dimension == 0) {
      return 0;
    }

    if (result > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("Tensor shape causes element-count overflow");
    }

    result *= dimension;
  }

  return result;
}

Tensor cuda_tensor_from_data(std::vector<std::size_t> shape,
                             std::vector<float> values, int device_index) {
#ifndef TENSORMORPH_HAS_CUDA
  (void)shape;
  (void)values;
  (void)device_index;

  throw std::runtime_error("TensorMorph was built without CUDA support");
#else
  if (device_index < 0) {
    throw std::invalid_argument("CUDA device index cannot be negative");
  }

  const std::size_t count = Tensor::checked_numel(shape);

  if (values.size() != count) {
    throw std::invalid_argument("Tensor data size does not match shape");
  }

  cuda::DeviceBuffer buffer(count * sizeof(float), device_index);

  if (count != 0) {
    cuda::copy_host_to_device(buffer.data(), values.data(),
                              count * sizeof(float), device_index);
  }

  return Tensor(std::move(shape),
                Tensor::Storage{Tensor::CudaStorage{std::move(buffer)}});
#endif
}

} // namespace tensormorph
