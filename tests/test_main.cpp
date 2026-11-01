#include "tensormorph/backend.hpp"
#include "tensormorph/operations.hpp"
#include "tensormorph/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void check_near_correlation(float actual, float expected,
                            float tolerance = 1.0e-5F) {
  // in the check, guarding againt NaN bug, coz, every NaN comparison results in
  // false if not like this and was if |act-exp| > tolerance, NaN will always
  // result in false and block doesn't execute
  if (!(std::fabs(actual - expected) < tolerance)) {
    throw std::runtime_error("Expected " + std::to_string(expected) +
                             ", received " + std::to_string(actual));
  }
}

// instead of checking one number, check all tensors
void check_values(const tensormorph::Tensor &tensor,
                  const std::vector<float> &expected,
                  float tolerance = 1.0e-5F) {
  check(tensor.numel() == expected.size(), "Element count mismatch");

  for (std::size_t i = 0; i < expected.size(); ++i) {
    check_near_correlation(tensor.data()[i], expected[i], tolerance);
  }
}

void check_throws(const std::function<void()> &function) {
  bool threw = false;

  try {
    function();
  } catch (const std::exception &) {
    threw = true;
  }

  check(threw, "Expected an exception");
}

void run_test(const std::string &name, const std::function<void()> &test) {
  try {
    test();
    ++passed;
    std::cout << "[PASS] " << name << '\n';
  } catch (const std::exception &error) {
    ++failed;
    std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
  }
}

using tensormorph::Tensor;

void test_tensor_shape_and_storage() {
  Tensor tensor({2, 3});

  check(tensor.rank() == 2, "Incorrect rank");
  check(tensor.numel() == 6, "Incorrect element count");
  check(tensor.shape()[0] == 2, "Incorrect first dimension");
  check(tensor.shape()[1] == 3, "Incorrect second dimension");

  for (float value : tensor.data()) {
    check_near_correlation(value, 0.0F);
  }
}

void test_scalar_and_empty_tensors() {
  Tensor scalar({});
  Tensor empty({0, 5});

  check(scalar.rank() == 0, "Scalar rank should be zero");
  check(scalar.numel() == 1, "Scalar should contain one element");

  check(empty.rank() == 2, "Empty tensor rank mismatch");
  check(empty.numel() == 0, "Empty tensor should have zero elements");
}

void test_tensor_data_validation() {
  check_throws([] { (void)Tensor::from_data({2, 2}, {1.0F, 2.0F, 3.0F}); });
}

void test_tensor_overflow() {
  check_throws(
      [] { (void)Tensor({std::numeric_limits<std::size_t>::max(), 2}); });
}

void test_tensor_copy_independence() {
  Tensor original = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});

  Tensor copied = original;
  copied.data()[0] = 99.0F;

  check_near_correlation(original.data()[0], 1.0F);
  check_near_correlation(copied.data()[0], 99.0F);
}

void test_tensor_move() {
  Tensor original = Tensor::from_data({3}, {1.0F, 2.0F, 3.0F});

  Tensor moved = std::move(original);

  check(moved.numel() == 3, "Moved tensor size mismatch");
  check_values(moved, {1.0F, 2.0F, 3.0F});
}

void test_addition() {
  Tensor a = Tensor::from_data({4}, {1, 2, 3, 4});
  Tensor b = Tensor::from_data({4}, {5, 6, 7, 8});

  Tensor result = tensormorph::add(a, b);

  check_values(result, {6, 8, 10, 12});
}

void test_addition_irregular_size() {
  Tensor a = Tensor::from_data({7}, {1, 2, 3, 4, 5, 6, 7});
  Tensor b = Tensor::from_data({7}, {7, 6, 5, 4, 3, 2, 1});

  check_values(tensormorph::add(a, b), {8, 8, 8, 8, 8, 8, 8});
}

void test_addition_in_place() {
  Tensor a = Tensor::from_data({3}, {1, 2, 3});
  Tensor b = Tensor::from_data({3}, {10, 20, 30});

  tensormorph::add_into(a, b, a);

  check_values(a, {11, 22, 33});
}

void test_addition_shape_mismatch() {
  Tensor a({2});
  Tensor b({3});

  check_throws([&] { (void)tensormorph::add(a, b); });
}

void test_multiplication() {
  Tensor a = Tensor::from_data({4}, {2, 3, 4, 5});
  Tensor b = Tensor::from_data({4}, {5, 4, 3, 2});

  check_values(tensormorph::multiply(a, b), {10, 12, 12, 10});
}

void test_multiplication_in_place() {
  Tensor a = Tensor::from_data({3}, {2, 3, 4});
  Tensor b = Tensor::from_data({3}, {5, 6, 7});

  tensormorph::multiply_into(a, b, a);

  check_values(a, {10, 18, 28});
}

void test_scale() {
  Tensor input = Tensor::from_data({4}, {1, -2, 3, -4});

  check_values(tensormorph::scale(input, -2.0F), {-2, 4, -6, 8});
}

void test_relu() {
  Tensor input = Tensor::from_data({5}, {-2, -0.0F, 0.0F, 3, 7});

  check_values(tensormorph::relu(input), {0, 0, 0, 3, 7});
}

void test_relu_in_place() {
  Tensor input = Tensor::from_data({4}, {-4, -1, 2, 5});

  tensormorph::relu_into(input, input);

  check_values(input, {0, 0, 2, 5});
}

void test_empty_elementwise_operations() {
  Tensor a({0, 4});
  Tensor b({0, 4});

  check(tensormorph::add(a, b).numel() == 0, "Empty addition failed");

  check(tensormorph::multiply(a, b).numel() == 0,
        "Empty multiplication failed");

  check(tensormorph::relu(a).numel() == 0, "Empty ReLU failed");
}

void test_reduction() {
  Tensor input = Tensor::from_data({5}, {1, 2, 3, 4, 5});

  check_near_correlation(tensormorph::reduce_sum(input), 15.0F);
}

void test_reduction_negative_values() {
  Tensor input = Tensor::from_data({4}, {-5, -3, 2, 1});

  check_near_correlation(tensormorph::reduce_sum(input), -5.0F);
}

void test_empty_reduction() {
  Tensor input({0});

  check_near_correlation(tensormorph::reduce_sum(input), 0.0F);
}

void test_matrix_multiplication() {
  Tensor a = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor b = Tensor::from_data({3, 2}, {7, 8, 9, 10, 11, 12});

  Tensor result = tensormorph::matmul(a, b);

  check(result.shape()[0] == 2, "Matmul rows mismatch");
  check(result.shape()[1] == 2, "Matmul columns mismatch");

  check_values(result, {58, 64, 139, 154});
}

void test_blocked_matrix_multiplication() {
  Tensor a = Tensor::from_data({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor b = Tensor::from_data({3, 2}, {7, 8, 9, 10, 11, 12});

  Tensor result = tensormorph::matmul_blocked(a, b, 2);

  check_values(result, {58, 64, 139, 154});
}

void test_blocked_matrix_multiplication_irregular() {
  Tensor a = Tensor::from_data(
      {3, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  Tensor b =
      Tensor::from_data({5, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});

  const Tensor reference = tensormorph::matmul(a, b);
  const Tensor blocked = tensormorph::matmul_blocked(a, b, 2);

  check_values(blocked,
               {175, 190, 205, 220, 400, 440, 480, 520, 625, 690, 755, 820});

  check_values(blocked, std::vector<float>(reference.data().begin(),
                                           reference.data().end()));
}

void test_matrix_shape_validation() {
  Tensor a({2, 3});
  Tensor b({4, 2});

  check_throws([&] { (void)tensormorph::matmul(a, b); });
}

void test_matrix_rank_validation() {
  Tensor a({6});
  Tensor b({2, 3});

  check_throws([&] { (void)tensormorph::matmul(a, b); });
}

void test_matrix_zero_block_size() {
  Tensor a({2, 2});
  Tensor b({2, 2});

  check_throws([&] { (void)tensormorph::matmul_blocked(a, b, 0); });
}

void test_zero_dimension_matrix() {
  Tensor a({2, 0});
  Tensor b({0, 3});

  Tensor result = tensormorph::matmul(a, b);

  check(result.shape()[0] == 2, "Zero-K matrix row mismatch");
  check(result.shape()[1] == 3, "Zero-K matrix column mismatch");

  check_values(result, {0, 0, 0, 0, 0, 0});
}

} // namespace

int main() {
  using Test = std::pair<const char *, std::function<void()>>;

  const std::vector<Test> tests = {
      {"tensor shape and storage", test_tensor_shape_and_storage},
      {"scalar and empty tensors", test_scalar_and_empty_tensors},
      {"tensor data validation", test_tensor_data_validation},
      {"tensor overflow", test_tensor_overflow},
      {"tensor copy independence", test_tensor_copy_independence},
      {"tensor move", test_tensor_move},
      {"addition", test_addition},
      {"addition irregular size", test_addition_irregular_size},
      {"addition in-place", test_addition_in_place},
      {"addition shape mismatch", test_addition_shape_mismatch},
      {"multiplication", test_multiplication},
      {"multiplication in-place", test_multiplication_in_place},
      {"scale", test_scale},
      {"ReLU", test_relu},
      {"ReLU in-place", test_relu_in_place},
      {"empty elementwise operations", test_empty_elementwise_operations},
      {"reduction", test_reduction},
      {"reduction negative values", test_reduction_negative_values},
      {"empty reduction", test_empty_reduction},
      {"matrix multiplication", test_matrix_multiplication},
      {"blocked matrix multiplication", test_blocked_matrix_multiplication},
      {"blocked matrix multiplication irregular",
       test_blocked_matrix_multiplication_irregular},
      {"matrix shape validation", test_matrix_shape_validation},
      {"matrix rank validation", test_matrix_rank_validation},
      {"zero matrix block size", test_matrix_zero_block_size},
      {"zero-dimension matrix", test_zero_dimension_matrix}};

  for (const auto &[name, test] : tests) {
    run_test(name, test);
  }

  std::cout << "\nResults: " << passed << " passed, " << failed << " failed\n";

  return failed == 0 ? 0 : 1;
}
