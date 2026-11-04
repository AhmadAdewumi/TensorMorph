
#include "../include/tensormorph/cuda/memory.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
  constexpr std::size_t count = 8;
  constexpr std::size_t bytes = count * sizeof(float);

  const std::vector<float> host_input{1.0F, 2.0F, 3.0F, 4.0F,
                                      5.0F, 6.0F, 7.0F, 8.0F};
  
  std::vector<float> host_output(count, 0.0F);

  tensormorph::cuda::DeviceBuffer device_buffer(bytes);

  tensormorph::cuda::copy_host_to_device(device_buffer.data(),
                                         host_input.data(), bytes);

  tensormorph::cuda::copy_device_to_host(host_input.data(), host_output.data(),
                                         bytes);

  for (std::size_t i{0}; i < count; ++i) {
    if (host_input[i] != host_output[i]) {
      std::cerr << "Memory transfer test validation failed" << "\n";
      return 1;
    }
  }

  std::cout << "GPU allocation and transfers test passed \n";
}
