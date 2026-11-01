#include "../include/tensormorph/tensor.hpp"
#include "tensormorph/operations.hpp"
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace tensormorph {

namespace {
void require_same_shape(const Tensor &a, const Tensor &b, const char *message) {
  if (!std::ranges::equal(a.shape(), b.shape())) {
    throw std::invalid_argument(message);
  }
}

void require_output_shape(const Tensor &input, const Tensor &output) {
  if (!std::ranges::equal(input.shape(), output.shape())) {
    throw std::invalid_argument("Output shape must match input shape");
  }
}
} // namespace

void add_into(const Tensor &a, const Tensor &b, Tensor &output) {
  require_same_shape(a, b, "Addition requires identical input shapes");
  require_output_shape(a, b);

  const auto a_data = a.data();
  const auto b_data = b.data();
  auto output_data = output.data();

  for (std::size_t i = 0; i < a_data.size(); ++i) {
    output_data[i] = a_data[i] + b_data[i];
  }
}

Tensor add(const Tensor &a, const Tensor &b) {
  require_same_shape(a, b, "Addition requires identical input shapes");

  Tensor output(std::vector<std::size_t>(a.shape().begin(), a.shape().end()));

  add_into(a, b, output);
  return output;
}

void multiply_into(const Tensor &a, const Tensor &b, Tensor &output) {
  require_same_shape(a, b, "multiplication requires identical input shapes");
  require_output_shape(a, b);

  const auto a_data = a.data();
  const auto b_data = b.data();
  auto output_data = output.data();

  for (std::size_t i = 0; i < a_data.size(); ++i) {
    output_data[i] = a_data[i] * b_data[i];
  }
}

Tensor multiply(const Tensor &a, const Tensor &b) {
  require_same_shape(a, b, "Addition requires identical input shapes");

  Tensor output(std::vector<std::size_t>(a.shape().begin(), a.shape().end()));

  multiply_into(a, b, output);
  return output;
}

void scale_into(const Tensor &input, float factor, Tensor &output) {
  require_output_shape(input, output);

  const auto src = input.data();
  auto dst = output.data();

  for (std::size_t i{0}; i < src.size(); ++i) {
    dst[i] = src[i] * factor;
  }
}

Tensor scale(const Tensor &input, float factor) {
  Tensor output(
      std::vector<std::size_t>(input.shape().begin(), input.shape().end()));

  scale_into(input, factor, output);
  return output;
}

void relu_into(const Tensor &input, Tensor &output) {
  const auto src = input.data();
  auto dst = output.data();

  for (std::size_t i{0}; i < src.size(); ++i) {
    dst[i] = std::max(0.0f, src[i]);
  }
}

Tensor relu(const Tensor &input) {
  Tensor output(
      std::vector<std::size_t>(input.shape().begin(), input.shape().end()));

  relu_into(input, output);
  return output;
}
} // namespace tensormorph
