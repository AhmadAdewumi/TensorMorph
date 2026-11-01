#pragma once

#include "tensormorph/tensor.hpp"

namespace tensormorph {

// allocates a nnew tensor and return elementwise a+b
Tensor add(const Tensor &a, const Tensor &b);

// we write the result into an existing tensor
// // output may alias either of the input
void add_into(const Tensor &a, const Tensor &b, Tensor &output);

[[nodiscard]] Tensor multiply(const Tensor &a, const Tensor &b);
void multiply_into(const Tensor &a, const Tensor &b, Tensor &output);

[[nodiscard]] Tensor scale(const Tensor &input, float factor);
void scale_into(const Tensor &input, float factor, Tensor &output);

[[nodiscard]] Tensor relu(const Tensor &input);
void relu_into(const Tensor &input, Tensor &output);

[[nodiscard]] float reduce_sum(const Tensor &input);

[[nodiscard]] Tensor matmul(const Tensor &a, const Tensor &b);
[[nodiscard]] Tensor matmul_blocked(const Tensor &a, const Tensor &b,
                                    std::size_t block_size = 32);
} // namespace tensormorph
