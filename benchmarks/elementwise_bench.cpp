#include "tensormorph/operations.hpp"
#include "tensormorph/tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using Clock = std::chrono::steady_clock;

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void benchmark(std::size_t count, int repetitions) {
    using tensormorph::Tensor;

    Tensor a({count});
    Tensor b({count});
    Tensor output({count});

    std::fill(a.data().begin(), a.data().end(), 1.25F);
    std::fill(b.data().begin(), b.data().end(), 2.75F);

    for (int i = 0; i < 3; ++i) {
        tensormorph::add_into(a, b, output);
    }

    std::vector<double> samples;
    samples.reserve( 9 );

    for (int sample = 0; sample < 9; ++sample) {
        const auto start = Clock::now();

        for (int repetition = 0; repetition < repetitions; ++repetition) {
            tensormorph::add_into(a, b, output);
        }

        const auto end = Clock::now();

        const double elapsed =
            std::chrono::duration<double>(end - start).count();

        samples.push_back(elapsed / repetitions);
    }

    const double seconds = median(samples);
    const double ns_per_element =
        seconds * 1.0e9 / static_cast<double>(count);

    const double effective_gb_s =
        (12.0 * static_cast<double>(count)) /
        seconds / 1.0e9;

    const double checksum = std::accumulate(
        output.data().begin(),
        output.data().end(),
        0.0
    );

    std::cout << std::setw(12) << count
              << std::setw(12) << repetitions
              << std::setw(16) << std::fixed << std::setprecision(2)
              << seconds * 1.0e6
              << std::setw(16) << ns_per_element
              << std::setw(14) << effective_gb_s
              << " checksum=" << checksum
              << '\n';
}

int main() {
    std::cout << "TensorMorph CPU elementwise addition\n";
    std::cout << "Preallocated inputs and output; median of 9 samples\n\n";

    std::cout << std::setw(12) << "Elements"
              << std::setw(12) << "Repeats"
              << std::setw(16) << "us/op"
              << std::setw(16) << "ns/element"
              << std::setw(14) << "GB/s"
              << '\n';

    benchmark(1024, 10000);
    benchmark(1U << 20U, 100);
    benchmark(1U << 24U, 10);
}