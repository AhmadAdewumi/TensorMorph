#include "tensormorph/operations.hpp"
#include "tensormorph/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tensormorph {

void add_into(const Tensor &a, const Tensor &b, Tensor &output) {
  if (!std::ranges::equal(a.shape(), b.shape())) {
    throw std::invalid_argument(
        "Addition of tensors requires them to have same shape");
  }

  if (!std::ranges::equal(a.shape(), output.shape())) {
    throw std::invalid_argument("Output shape must match input shape");
  }

  const auto a_data = a.data();
  const auto b_data = b.data();
  const auto result = output.data();

  for (std::size_t i = 0; i < a_data.size(); ++i) {
    result[i] = a_data[i] + b_data[i];
  }
}

Tensor add(const Tensor &a, const Tensor &b) {
  if (!std::ranges::equal(a.shape(), b.shape())) {
    throw std::invalid_argument(
        "Addition of tensors requires them to have same shape");
  }

  Tensor output(std::vector<std::size_t>(a.shape().begin(), a.shape().end()));

  add_into(a, b, output);

  return output;
}
} // namespace tensormorph
