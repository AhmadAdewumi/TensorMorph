
#include "tensormorph/tensor.hpp"
#include <cstddef>
#include <stdexcept>
namespace tensormorph {
namespace {
struct MatrixDimensions {
  std::size_t m;
  std::size_t k;
  std::size_t n;
};

MatrixDimensions validate_matmul(const Tensor &a, const Tensor &b) {
  if (a.rank() != 2 || b.rank() != 2) {
    throw std::invalid_argument("Mat mul requires rank-2 tensors");
  }

  const auto a_shape = a.shape();
  const auto b_shape = b.shape();

  const std::size_t m = a_shape[0];
  const std::size_t k = a_shape[1];
  const std::size_t b_k = b_shape[0];
  const std::size_t n = b_shape[1];

  if (k != b_k) {
    throw std::invalid_argument(
        "Matrix multiplication dimension are incompatible");
  }

  return {m, k, n};
}

void reject_output_alias(const Tensor& a, const Tensor& b, const Tensor& output){
    if (&output == &a || & output == &b) {
        throw std::invalid_argument("Matmul cannot alias an output");    
    }
    
}
} // namespace


Tensor matmul(const Tensor& a, const Tensor& b) {
    const auto dimensions = validate_matmul(a, b);

    Tensor output({dimensions.m, dimensions.n});

    const auto a_data = a.data();
    const auto b_data = b.data();
    auto dst = output.data();

    for (std::size_t row = 0; row < dimensions.m; ++row) {
        for (std::size_t col = 0; col < dimensions.n; ++col) {
            float sum = 0.0F;

            for (std::size_t inner = 0; inner < dimensions.k; ++inner) {
                sum += a_data[row * dimensions.k + inner] *
                       b_data[inner * dimensions.n + col];
            }

            dst[row * dimensions.n + col] = sum;
        }
    }

    return output;
}

Tensor matmul_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size
) {
    const auto dimensions = validate_matmul(a, b);

    if (block_size == 0) {
        throw std::invalid_argument(
            "Matrix multiplication block size must be positive"
        );
    }

    Tensor output({dimensions.m, dimensions.n});

    const auto a_data = a.data();
    const auto b_data = b.data();
    auto destination = output.data();

    // The output was zero-initialized by Tensor's constructor.
    for (std::size_t ii = 0; ii < dimensions.m; ii += block_size) {
        for (std::size_t kk = 0; kk < dimensions.k; kk += block_size) {
            for (std::size_t jj = 0; jj < dimensions.n; jj += block_size) {
                const std::size_t i_end =
                    std::min(ii + block_size, dimensions.m);
                const std::size_t k_end =
                    std::min(kk + block_size, dimensions.k);
                const std::size_t j_end =
                    std::min(jj + block_size, dimensions.n);

                for (std::size_t row = ii; row < i_end; ++row) {
                    for (std::size_t inner = kk; inner < k_end; ++inner) {
                        const float a_value =
                            a_data[row * dimensions.k + inner];

                        for (std::size_t col = jj; col < j_end; ++col) {
                            destination[row * dimensions.n + col] +=
                                a_value *
                                b_data[inner * dimensions.n + col];
                        }
                    }
                }
            }
        }
    }

    return output;
}

} // namespace tensormorph
