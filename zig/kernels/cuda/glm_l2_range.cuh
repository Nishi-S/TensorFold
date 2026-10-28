// Byte windows wholly inside a borrowed allocation, shared with the host boundary proof.
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define TF_L2_HD __host__ __device__ __forceinline__
#else
#define TF_L2_HD inline
#endif
namespace tf_glm_l2 {
struct Window { uint64_t start, bytes; };
TF_L2_HD Window aligned(uint64_t address, uint64_t bytes, uint64_t alignment) {
    const uint64_t start = (uint64_t(0) - address) & (alignment - 1);
    return {start, bytes > start ? (bytes - start) & ~(alignment - 1) : 0};
}
} // namespace tf_glm_l2
#undef TF_L2_HD
