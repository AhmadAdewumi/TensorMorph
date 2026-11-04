
#include <cstddef>
#include <string>
namespace tensormorph::cuda {
struct DeviceInfo {
  int id = 0;
  std::string name;

  int compute_major = 0;
  int compute_minor = 0;

  int multiprocessor_count = 0;
  int warp_size = 0;

  std::size_t total_global_memory = 0;
  std::size_t shared_memory_per_block = 0;

  int registers_per_block = 0;
  int async_engine_count = 0;
  int concurrent_kernels = 0;
};

[[nodiscard]]
int device_count();

// returns info a boaut a cude device
[[nodiscard]]
DeviceInfo device_info(int device_index = 0);

// select the CUDA device for calling the host thread
void set_device(int device_index);

// synchronize the selected device
void synchronize(int device_index = 0);
} // namespace tensormorph::cuda
