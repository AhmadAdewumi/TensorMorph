
#include <cstddef>
namespace tensormorph::cuda {
class DeviceBuffer {
private:
  // where is the allocation
  void *pointer_ = nullptr;
  // how large is the allocation
  std::size_t size_bytes_ = 0;
  int device_index_ = 0;

public:
  explicit DeviceBuffer(std::size_t bytes, int device_index = 0);

  ~DeviceBuffer() noexcept;

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  [[nodiscard]]
  void *data() noexcept;

  [[nodiscard]]
  const void *data() const noexcept;

  [[nodiscard]]
  std::size_t size_bytes() const noexcept;

  [[nodiscard]]
  int device_index() const noexcept;

  [[nodiscard]]
  explicit operator bool() const noexcept;

  void reset() noexcept;
};

void copy_host_to_device(void *destination, const void *source,
                         std::size_t bytes, int device_index = 0);
void copy_device_to_host(void *destination, const void *source,
                         std::size_t bytes, int device_index = 0);
void copy_device_to_device(void *destination, const void *source,
                           std::size_t bytes, int device_index = 0);
} // namespace tensormorph::cuda
