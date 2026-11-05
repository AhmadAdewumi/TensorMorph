#pragma once

#include "tensormorph/tensor.hpp"

namespace tensormorph::cpu {

Tensor add(const Tensor &a, const Tensor &b);
void add_into(const Tensor &a, const Tensor &b, Tensor &output);

Tensor multiply(const Tensor &a, const Tensor &b);
void multiply_into(const Tensor &a, const Tensor &b, Tensor &output);

Tensor scale(const Tensor &input, float factor);
void scale_into(const Tensor &input, float factor, Tensor &output);

Tensor relu(const Tensor &input);
void relu_into(const Tensor &input, Tensor &output);

float reduce_sum(const Tensor &input);

Tensor matmul(const Tensor &a, const Tensor &b);
Tensor matmul_blocked(const Tensor &a, const Tensor &b, std::size_t block_size);

} // namespace tensormorph::cpu
