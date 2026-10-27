// Every E4M3FN value is exact in BF16; the two NaN encodings become signed quiet BF16 NaNs.
#pragma once
#include <stdint.h>

namespace tf_glm_fp8 {
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint16_t bf16(uint8_t code) {
    const uint16_t sign = (uint16_t)(code & 128u) << 8;
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return sign | 0x7fc0u;
    if (exponent != 0) return sign | (uint16_t)((exponent + 120) << 7) | (uint16_t)(mantissa << 4);
    if (mantissa == 0) return sign;
    const int high = mantissa >= 4 ? 2 : mantissa >= 2 ? 1 : 0;
    return sign | (uint16_t)((118 + high) << 7) | (uint16_t)((mantissa << (7 - high)) & 127);
}
} // namespace tf_glm_fp8
