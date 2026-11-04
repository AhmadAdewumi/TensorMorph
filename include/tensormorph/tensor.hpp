#pragma once

#include "tensormorph/cuda/memory.hpp"
#include "tensormorph/device.hpp"
#include <cstddef>
#include <span>
#include <variant>
#include <vector>

namespace tensormorph {

class Tensor {
public:
  explicit Tensor(std::vector<std::size_t> shape,
                  Device device = Device::cpu());

  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);

  Tensor(Tensor &&other) = default;
  Tensor &operator=(Tensor &&other) noexcept = default;

  ~Tensor() = default;

  // construct a tensor from existing values, while validating thier count also
  // should consider seapping this out for a constructor, ain't this kinda
  // readable or what
  static Tensor from_data(std::vector<std::size_t> shape,
                          std::vector<float> values,
                          Device device = Device::cpu());

  [[nodiscard]]
  Device device() const noexcept;

  [[nodiscard]]
  Tensor to(Device target) const;

  [[nodiscard]]
  std::span<const std::size_t> shape() const noexcept;

  /*
   * returns the no od dim of the tensor
   * rank: 0 -> scalar, 1->vector, 2-> matrix, 3+ -> multi-dim arr
   */
  [[nodiscard]]
  std::size_t rank() const noexcept;

  /*
   * returns the total number of individaual values in the tensor
   * (product of all the dim in the tensor shape)
   */
  [[nodiscard]]
  std::size_t numel() const noexcept;

  [[nodiscard]]
  std::span<float> data();

  std::span<const float> data() const;

private:
  struct CpuStorage {
    std::vector<float> values;
  };

  struct CudaStorage {
    cuda::DeviceBuffer buffer;
  };

  using Storage = std::variant<CpuStorage, CudaStorage>;

  Tensor(std::vector<std::size_t> shape, Storage storage);

  static std::size_t checked_numel(const std::vector<std::size_t> &shape);

  [[nodiscard]] void *device_data();
  [[nodiscard]] const void *device_data() const;

  std::vector<std::size_t> shape_;
  std::size_t numel_;
  Storage storage_;

  friend Tensor cuda_tensor_from_data(std::vector<std::size_t>,
                                      std::vector<float>, int device_index);
};

} // namespace tensormorph
