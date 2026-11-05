
#include "tensormorph/tensor.hpp"
#include "tensormorph/operations.hpp"
#include "tensormorph/cuda/runtime.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace tensormorph;

int main() {
    if (cuda::device_count() == 0) {
        std::cout << "No CUDA device. Skipping.\n";
        return 0;
    }

    constexpr std::size_t count = 1024 * 1024;

    std::vector<float> left(count, 2.0F);
    std::vector<float> right(count, 3.0F);

    Tensor a = Tensor::from_data({count}, std::move(left));
    Tensor b = Tensor::from_data({count}, std::move(right));

    Tensor ga = a.to(Device::gpu());
    Tensor gb = b.to(Device::gpu());

    Tensor gc = add(ga, gb);
    Tensor gd = multiply(gc, gc);
    Tensor ge = scale(gd, 0.5F);
    Tensor gf = relu(ge);

    Tensor result = gf.to(Device::cpu());

    for (const float value : result.data()) {
        if (std::abs(value - 12.5F) > 1e-5F) {
            throw std::runtime_error("CUDA elementwise result incorrect");
        }
    }

    const float sum = reduce_sum(ga);

    const float expected =
        static_cast<float>(count) * 2.0F;

    if (std::abs(sum - expected) > 1.0F) {
        throw std::runtime_error("CUDA reduction incorrect");
    }

    std::cout << "CUDA operations passed.\n";
    return 0;
}