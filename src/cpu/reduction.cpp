#include "../include/tensormorph/operations.hpp"

namespace tensormorph {

float reduce_sum(const Tensor& input) {
    const auto values = input.data();

    float sum = 0.0F;

    for (const float value : values) {
        sum += value;
    }

    return sum;
}

} // namespace tensormorph