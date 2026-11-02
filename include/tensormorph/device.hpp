

namespace tensormorph {
enum class DeviceType { CPU, CUDA };

struct Device {
  DeviceType type = DeviceType::CPU;
  int index = 0;

  [[nodiscard]]
  constexpr bool is_cpu() const noexcept {
    return type == DeviceType::CPU;
  }

  [[nodiscard]]
  constexpr bool is_gpu() const noexcept {
    return type == DeviceType::CUDA;
  }

  [[nodiscard]]
  static constexpr Device cpu() {
    return {DeviceType::CPU, 0};
  }

  [[nodiscard]]
  static constexpr Device gpu(int index = 0) {
    return {DeviceType::CUDA, index};
  }

  friend constexpr bool operator==(const Device &lhs, const Device &rhs) {
    return lhs.type == rhs.type && lhs.index == rhs.index;
  }
};

} // namespace tensormorph
