#pragma once

#include "tensormorph/tensor.hpp"
#include <cstddef>
#include <vector>

namespace tensormorph {
Tensor cuda_tensor_from_data(std::vector<std::size_t> shape,
                             std::vector<float> values, int device_index = 0);
}
