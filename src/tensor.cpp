#include "../include/tensormorph/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tensormorph {
// this calculates the number of element the tensor needs
std::size_t Tensor::checked_numel(const std::vector<std::size_t> &shape) {
  // a rank-zero tesnsor is a scalar
  std::size_t count = 1;

  for (const std::size_t dimension : shape) {
    // coz say we have {2,0,3}, multiplying by zero makes the tensor empty
    if (dimension == 0) {
      return 0;
    }

    // check to prevent unsigned int overflow
    if (count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::length_error("Tensor element count overflows mighty size_t");
    }

    count *= dimension;
  }

  return count;
}

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
  if (data_.size() != checked_numel(shape_)) {
    throw std::invalid_argument("Tensor data size does not match");
  }
}

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(checked_numel(shape_), 0.0f) {}

Tensor Tensor::from_data(std::vector<std::size_t> shape,
                         std::vector<float> values) {

  return Tensor(std::move(shape), std::move(values));
}

[[nodiscard]]
std::span<const std::size_t> Tensor::shape() const noexcept {
  return shape_;
};

[[nodiscard]]
std::size_t Tensor::rank() const noexcept {
  return shape_.size();
};

[[nodiscard]]
std::size_t Tensor::numel() const noexcept {
  return data_.size();
};

[[nodiscard]]
std::span<float> Tensor::data() noexcept {
  return data_;
};

std::span<const float> Tensor::data() const noexcept { return data_; };

} // namespace tensormorph
