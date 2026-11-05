// #include "tensormorph/operations.hpp"
// #include "tensormorph/tensor.hpp"
// #include <algorithm>
// #include <cstddef>
// #include <stdexcept>
// #include <vector>
// 
// namespace tensormorph {
// 
// void add_into(const Tensor &a, const Tensor &b, Tensor &output) {
//   if (!std::ranges::equal(a.shape(), b.shape())) {
//     throw std::invalid_argument(
//         "Addition of tensors requires them to have same shape");
//   }
// 
//   if (!std::ranges::equal(a.shape(), output.shape())) {
//     throw std::invalid_argument("Output shape must match input shape");
//   }
// 
//   const auto a_data = a.data();
//   const auto b_data = b.data();
//   const auto result = output.data();
// 
//   for (std::size_t i = 0; i < a_data.size(); ++i) {
//     result[i] = a_data[i] + b_data[i];
//   }
// }
// 
// Tensor add(const Tensor &a, const Tensor &b) {
//   if (!std::ranges::equal(a.shape(), b.shape())) {
//     throw std::invalid_argument(
//         "Addition of tensors requires them to have same shape");
//   }
// 
//   Tensor output(std::vector<std::size_t>(a.shape().begin(), a.shape().end()));
// 
//   add_into(a, b, output);
// 
//   return output;
// }
// } // namespace tensormorph


#include "tensormorph/operations.hpp"
#include "cpu_ops.hpp"

#ifdef TENSORMORPH_HAS_CUDA
#include "tensormorph/cuda/kernels.hpp"
#endif

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace tensormorph {

namespace {

void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != b.shape().size() ||
        !std::equal(a.shape().begin(), a.shape().end(), b.shape().begin())) {
        throw std::invalid_argument("Tensor shapes do not match");
    }
}

void require_same_device(const Tensor& a, const Tensor& b) {
    if (a.device() != b.device()) {
        throw std::invalid_argument("Tensor devices do not match");
    }
}

Tensor make_output(const Tensor& input) {
    return Tensor(
        std::vector<std::size_t>(
            input.shape().begin(), input.shape().end()
        ),
        input.device()
    );
}

}

Tensor add(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b);
    require_same_device(a, b);

    if (a.device().is_cpu()) {
        return cpu::add(a, b);
    }

#ifdef TENSORMORPH_HAS_CUDA
    Tensor output = make_output(a);

    cuda::kernels::add(
        static_cast<const float*>(a.device_data()),
        static_cast<const float*>(b.device_data()),
        static_cast<float*>(output.device_data()),
        a.numel(),
        a.device().index
    );

    cuda::synchronize(a.device().index);
    return output;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

void add_into(const Tensor& a, const Tensor& b, Tensor& output) {
    require_same_shape(a, b);
    require_same_device(a, b);

    if (a.shape().size() != output.shape().size() ||
        !std::equal(a.shape().begin(), a.shape().end(), output.shape().begin())) {
        throw std::invalid_argument("Output shape mismatch");
    }

    require_same_device(a, output);

    if (a.device().is_cpu()) {
        cpu::add_into(a, b, output);
        return;
    }

#ifdef TENSORMORPH_HAS_CUDA
    cuda::kernels::add(
        static_cast<const float*>(a.device_data()),
        static_cast<const float*>(b.device_data()),
        static_cast<float*>(output.device_data()),
        a.numel(),
        a.device().index
    );
    cuda::synchronize(a.device().index);
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b);
    require_same_device(a, b);

    if (a.device().is_cpu()) {
        return cpu::multiply(a, b);
    }

#ifdef TENSORMORPH_HAS_CUDA
    Tensor output = make_output(a);

    cuda::kernels::multiply(
        static_cast<const float*>(a.device_data()),
        static_cast<const float*>(b.device_data()),
        static_cast<float*>(output.device_data()),
        a.numel(),
        a.device().index
    );

    cuda::synchronize(a.device().index);
    return output;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

void multiply_into(const Tensor& a, const Tensor& b, Tensor& output) {
    require_same_shape(a, b);
    require_same_device(a, b);

    if (a.shape().size() != output.shape().size() ||
        !std::equal(a.shape().begin(), a.shape().end(), output.shape().begin())) {
        throw std::invalid_argument("Output shape mismatch");
    }

    require_same_device(a, output);

    if (a.device().is_cpu()) {
        cpu::multiply_into(a, b, output);
        return;
    }

#ifdef TENSORMORPH_HAS_CUDA
    cuda::kernels::multiply(
        static_cast<const float*>(a.device_data()),
        static_cast<const float*>(b.device_data()),
        static_cast<float*>(output.device_data()),
        a.numel(),
        a.device().index
    );
    cuda::synchronize(a.device().index);
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

Tensor scale(const Tensor& input, float scalar) {
    if (input.device().is_cpu()) {
        return cpu::scale(input, scalar);
    }

#ifdef TENSORMORPH_HAS_CUDA
    Tensor output = make_output(input);

    cuda::kernels::scale(
        static_cast<const float*>(input.device_data()),
        scalar,
        static_cast<float*>(output.device_data()),
        input.numel(),
        input.device().index
    );

    cuda::synchronize(input.device().index);
    return output;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

void scale_into(const Tensor& input, float scalar, Tensor& output) {
    if (input.device() != output.device()) {
        throw std::invalid_argument("Input and output devices differ");
    }

    if (input.shape().size() != output.shape().size() ||
        !std::equal(input.shape().begin(), input.shape().end(), output.shape().begin())) {
        throw std::invalid_argument("Output shape mismatch");
    }

    if (input.device().is_cpu()) {
        cpu::scale_into(input, scalar, output);
        return;
    }

#ifdef TENSORMORPH_HAS_CUDA
    cuda::kernels::scale(
        static_cast<const float*>(input.device_data()),
        scalar,
        static_cast<float*>(output.device_data()),
        input.numel(),
        input.device().index
    );
    cuda::synchronize(input.device().index);
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

Tensor relu(const Tensor& input) {
    if (input.device().is_cpu()) {
        return cpu::relu(input);
    }

#ifdef TENSORMORPH_HAS_CUDA
    Tensor output = make_output(input);

    cuda::kernels::relu(
        static_cast<const float*>(input.device_data()),
        static_cast<float*>(output.device_data()),
        input.numel(),
        input.device().index
    );

    cuda::synchronize(input.device().index);
    return output;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

void relu_into(const Tensor& input, Tensor& output) {
    if (input.device() != output.device()) {
        throw std::invalid_argument("Input and output devices differ");
    }

    if (input.shape().size() != output.shape().size() ||
        !std::equal(input.shape().begin(), input.shape().end(), output.shape().begin())) {
        throw std::invalid_argument("Output shape mismatch");
    }

    if (input.device().is_cpu()) {
        cpu::relu_into(input, output);
        return;
    }

#ifdef TENSORMORPH_HAS_CUDA
    cuda::kernels::relu(
        static_cast<const float*>(input.device_data()),
        static_cast<float*>(output.device_data()),
        input.numel(),
        input.device().index
    );
    cuda::synchronize(input.device().index);
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

float reduce_sum(const Tensor& input) {
    if (input.device().is_cpu()) {
        return cpu::reduce_sum(input);
    }

#ifdef TENSORMORPH_HAS_CUDA
    cuda::DeviceBuffer result(sizeof(float), input.device().index);

    cuda::kernels::reduce_sum(
        static_cast<const float*>(input.device_data()),
        static_cast<float*>(result.data()),
        input.numel(),
        input.device().index
    );

    float host_result = 0.0F;

    cuda::copy_device_to_host(
        &host_result,
        result.data(),
        sizeof(float),
        input.device().index
    );

    return host_result;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("matmul requires rank-2 tensors");
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul dimensions are incompatible");
    }

    require_same_device(a, b);

    if (a.device().is_cpu()) {
        return cpu::matmul(a, b);
    }

#ifdef TENSORMORPH_HAS_CUDA
    Tensor output({a.shape()[0], b.shape()[1]}, a.device());

    cuda::kernels::matmul(
        static_cast<const float*>(a.device_data()),
        static_cast<const float*>(b.device_data()),
        static_cast<float*>(output.device_data()),
        a.shape()[0],
        a.shape()[1],
        b.shape()[1],
        a.device().index
    );

    cuda::synchronize(a.device().index);
    return output;
#else
    throw std::runtime_error("CUDA unavailable");
#endif
}

Tensor matmul_blocked(
    const Tensor& a,
    const Tensor& b,
    std::size_t block_size
) {
    if (block_size == 0) {
        throw std::invalid_argument("block_size must be positive");
    }

    require_same_device(a, b);

    if (a.device().is_cpu()) {
        return cpu::matmul_blocked(a, b, block_size);
    }

    // The first CUDA implementation uses the straightforward tiled-output
    // kernel. A shared-memory tiled GEMM is a separate optimization.
    return matmul(a, b);
}

}