
#include <cstddef>
namespace tensormorph::cuda::kernels {
void vector_add(const float *a, const float *b, float *output,
                std::size_t noOfElements);
}
