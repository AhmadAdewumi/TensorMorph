
#include <cstddef>
namespace tensormorph::cuda::kernels {
void vector_add(const float *a, const float *b, float *output,
                std::size_t noOfElements);

void add(const float *a, const float *b, float *output, std::size_t count,
         int device_index, void *stream = nullptr);

void multiply(const float *a, const float *b, float *output, std::size_t count,
              int device_index, void *stream = nullptr);
void scale(const float *input, float scalar, float *output, std::size_t count,
           int device_index, void *stream = nullptr);

void relu(const float *input, float *output, std::size_t count,
          int device_index, void *stream = nullptr);

void reduce_sum(const float *input, float *output, std::size_t count,
                int device_index, void *stream = nullptr);

void matmul(const float *a, const float *b, float *output, std::size_t m,
            std::size_t k, std::size_t n, int device_index,
            void *stream = nullptr);

void transpose(const float *input, float *output, std::size_t rows,
               std::size_t columns, int device_index, void *stream = nullptr);
} // namespace tensormorph::cuda::kernels
