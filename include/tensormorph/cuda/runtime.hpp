
#include <cstddef>
namespace tensormorph::cuda {
struct DeviceInfo {
  int id;
  char name[256];

  int compute_major;
  int compute_minor;
  int multiprocessor_count;
  int warp_size;
  std::size_t total_global_memory;
  std::size_t shared_memory_per_block;
  std::size_t registers_per_block;
  int async_engine_count;
  int concurrent_kernels;
};

[[nodiscard]]
int device_count();

// returns info a boaut a cude device
[[nodiscard]]
DeviceInfo device_info(int device_id);

// select the CUDA device for calling the host thread
void set_device(int device_id);

// synchronize the selected device
void synchronize();
} // namespace tensormorph::cuda
