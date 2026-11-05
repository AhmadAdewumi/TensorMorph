#include "../include/tensormorph/operations.hpp"
#include <stdexcept>

namespace tensormorph {

float reduce_sum(const Tensor &input) {
  if (!input.device().is_cpu()) {
    throw std::invalid_argument("CPU reduction received CUDA tensor");
  }
  const auto values = input.data();

  float sum = 0.0F;

  for (const float value : values) {
    sum += value;
  }

  return sum;
}

} // namespace tensormorph
