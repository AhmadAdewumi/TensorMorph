
#include "tensormorph/operations.hpp"
#include "tensormorph/tensor.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>
namespace {
using Clock = std::chrono::steady_clock;

double run_sample(const tensormorph::Tensor &a, const tensormorph::Tensor &b,
                  tensormorph::Tensor &output, std::size_t repititions) {
  const auto start = Clock::now();

  for (std::size_t i = 0; i < repititions; ++i) {
    tensormorph::add_into(a, b, output);
  }

  const auto stop = Clock::now();

  return std::chrono::duration<double>(stop - start).count();
}

void benchmark(std::size_t number_of_elements, std::size_t repititions) {
  auto a = tensormorph::Tensor({number_of_elements});
  auto b = tensormorph::Tensor({number_of_elements});
  auto output = tensormorph::Tensor({number_of_elements});

  // populating the tensors with values
  for (std::size_t i = 0; i < number_of_elements; ++i) {
    a.data()[i] = static_cast<float>(i % 100);
    b.data()[i] = static_cast<float>((i % 100) * 2);
  }

  // .. warm  up
  for (std::size_t i = 0; i < 3; ++i) {
    tensormorph::add_into(a, b, output);
  }

  // container to store the execution times
  constexpr std::size_t sample_count = 7;
  std::array<double, sample_count> samples{};

  for (double &sample : samples) {
    sample = run_sample(a, b, output, repititions);
  }

  std::sort(samples.begin(), samples.end());
  const double median_seconds = samples[sample_count / 2];

  const double seconds_per_operation =
      median_seconds / static_cast<double>(repititions);
  const double nanoseconds_per_element =
      seconds_per_operation * 1e9 / static_cast<double>(number_of_elements);

  // two float32 reads and one float32 write per operation
  const double bytes_per_operation = number_of_elements * sizeof(float) * 3;

  // bytes/sec / 10^9
  const double gb_per_second =
      (bytes_per_operation / seconds_per_operation) / 1e9;

  double checksum = 0.0;
  for (const float value : output.data()) {
    checksum += value;
  }

  std::cout << std::setw(10) << number_of_elements << std::setw(12)
            << repititions << std::setw(18) << std::fixed
            << std::setprecision(2) << seconds_per_operation * 1e6
            << std::setw(20) << nanoseconds_per_element << std::setw(18)
            << gb_per_second << "  checksum=" << checksum << '\n';
}
} // namespace

int main() {
  std::cout << "CPU vector-add benchmark: preallocated output\n"
            << "Times are medians of seven samples after three warmups.\n"
            << "Effective bandwidth assumes 12 bytes per element including 2 "
               "reads and a write.\n\n"
            << std::setw(10) << "Elements" << std::setw(12) << "Repeats"
            << std::setw(18) << "(micro) us/op" << std::setw(20) << "ns/element"
            << std::setw(18) << "GB/s" << '\n';

  benchmark(1024, 10'000);
  benchmark(1'048'576, 100);
  benchmark(16'777'216, 10);
}
