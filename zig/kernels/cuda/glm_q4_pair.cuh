// Adjacent unsigned nibbles as exact bf16 128+q pairs; subtracting 136 gives the symmetric Q4 codes.
#pragma once
#include <stdint.h>

namespace tf_glm_q4 {
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint32_t biased_pair(uint32_t word, int pair) {
    const uint32_t byte = word >> (8 * pair);
    return 0x43004300u | (byte & 15u) | ((byte & 240u) << 12);
}

#if defined(__CUDACC__)
__device__ __forceinline__ uint32_t decode_pair(uint32_t word, int pair) {
    const uint32_t packed = biased_pair(word, pair);
    uint32_t out;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(out) : "r"(packed), "r"(0x43084308u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(out) : "r"(packed), "r"(0x3f803f80u), "r"(0xc308c308u));
#endif
    return out;
}
#endif
} // namespace tf_glm_q4
