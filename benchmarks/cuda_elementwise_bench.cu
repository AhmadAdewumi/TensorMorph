#include "tensormorph/cuda/event.hpp"
#include "tensormorph/cuda/kernels.hpp"
#include "tensormorph/cuda/memory.hpp"
#include "tensormorph/cuda/runtime.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <vector>

int main() {
  using namespace tensormorph::cuda;

  if (device_count() == 0) {
    std::cerr << "No CUDA device available.\n";
    return 1;
  }

  constexpr std::size_t count = 16 * 1024 * 1024;
  constexpr int warmup = 10;
  constexpr int iterations = 100;

  const std::size_t bytes = count * sizeof(float);

  std::vector<float> host_a(count, 2.0F);
  std::vector<float> host_b(count, 3.0F);

  DeviceBuffer a(bytes);
  DeviceBuffer b(bytes);
  DeviceBuffer output(bytes);

  copy_host_to_device(a.data(), host_a.data(), bytes);
  copy_host_to_device(b.data(), host_b.data(), bytes);

  for (int i = 0; i < warmup; ++i) {
    kernels::add(static_cast<const float *>(a.data()),
                 static_cast<const float *>(b.data()),
                 static_cast<float *>(output.data()), count, 0);
  }

  Event start;
  Event stop;

  start.record();

  for (int i = 0; i < iterations; ++i) {
    kernels::add(static_cast<const float *>(a.data()),
                 static_cast<const float *>(b.data()),
                 static_cast<float *>(output.data()), count, 0);
  }

  stop.record();
  stop.synchronize();

  const float total_ms = start.elapsed_ms(stop);
  const double average_ms = total_ms / iterations;

  const double transferred_bytes =
      static_cast<double>(count) * sizeof(float) * 3.0;

  const double bandwidth_gbs = transferred_bytes / (average_ms * 1.0e6);

  std::cout << "Elements: " << count << '\n';
  std::cout << "Average kernel time: " << average_ms << " ms\n";
  std::cout << "Effective bandwidth: " << bandwidth_gbs << " GB/s\n";
}
