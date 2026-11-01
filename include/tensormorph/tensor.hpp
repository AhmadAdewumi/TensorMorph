#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace tensormorph {

class Tensor {
public:
  explicit Tensor(std::vector<std::size_t> shape);

  // construct a tensor from existing values, while validating thier count also
  // should consider seapping this out for a constructor, ain't this kinda
  // readable or what
  static Tensor from_data(std::vector<std::size_t> shape,
                          std::vector<float> values);

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
  std::span<float> data() noexcept;

  std::span<const float> data() const noexcept;

private:
  Tensor(std::vector<std::size_t> shape, std::vector<float> data);

  static std::size_t checked_numel(const std::vector<std::size_t> &shape);
  std::vector<std::size_t> shape_;
  std::vector<float> data_;
};

} // namespace tensormorph
