#include "tensormorph/operations.hpp"
#include "tensormorph/tensor.hpp"

#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;

template <typename Function> double measure(Function &&function) {
  const auto start = Clock::now();
  function();
  const auto end = Clock::now();

  return std::chrono::duration<double>(end - start).count();
}

float checksum(tensormorph::Tensor &tensor) {
  float sum = 0.0;
  for (float value : tensor.data()) {
    sum += value;
  }
  return sum;
}

int main() {
  using tensormorph::Tensor;

  constexpr std::size_t n = 256;

  Tensor a({n, n});
  Tensor b({n, n});

  for (std::size_t i = 0; i < a.numel(); ++i) {
    a.data()[i] = static_cast<float>(i % 17) / 17.0F;
    b.data()[i] = static_cast<float>(i % 13) / 13.0F;
  }

  // Warm-up.
  (void)tensormorph::matmul(a, b);
  (void)tensormorph::matmul_blocked(a, b, 32);

  constexpr int samples = 7;

  std::vector<double> naive_times;
  std::vector<double> blocked_times;

  for (int i = 0; i < samples; ++i) {
    naive_times.push_back(
        measure([&] { const auto result = tensormorph::matmul(a, b); }));

    blocked_times.push_back(measure(
        [&] { const auto result = tensormorph::matmul_blocked(a, b, 32); }));
  }

  const auto median = [](std::vector<double> &values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };

  const double naive = median(naive_times);
  const double blocked = median(blocked_times);

  // number of operations carried out, for a 2x2 matrix, 2 multiplication, 1
  // addition, 3x3, 3 mul, 2add for nxn matrix, n mul and n-1 addm so
  // n+(n-1)=2n-1m for nxn matrix, output elements = n^3 multiplying bot tho
  // get, 2n^3-n^2m we use 2n^3
  const double operations = 2.0 * static_cast<double>(n) *
                            static_cast<double>(n) * static_cast<double>(n);

  std::cout << "TensorMorph CPU matrix multiplication\n";
  std::cout << "Matrix dimensions: " << n << " x " << n << '\n';
  std::cout << std::fixed << std::setprecision(3);

  std::cout << "Naive:   " << naive * 1.0e3 << " ms, "
            << operations / naive / 1.0e9 << " GFLOP/s\n";

  std::cout << "Blocked: " << blocked * 1.0e3 << " ms, "
            << operations / blocked / 1.0e9 << " GFLOP/s\n";

  std::cout << "Speedup: " << naive / blocked << "x\n";
}
