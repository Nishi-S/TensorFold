// Exhaust the exact E4M3FN-to-BF16 bit conversion used by the GLM FP8 weight-only kernel.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include "../../../zig/kernels/cuda/glm_fp8_decode.cuh"

int main() {
    for (unsigned code = 0; code < 256; ++code) {
        const auto bf = tf_glm_fp8::bf16(static_cast<uint8_t>(code));
        const uint32_t bits = static_cast<uint32_t>(bf) << 16;
        float actual;
        std::memcpy(&actual, &bits, sizeof(actual));
        const int exponent = (code >> 3) & 15, mantissa = code & 7;
        if (exponent == 15 && mantissa == 7) {
            if (!std::isnan(actual) || std::signbit(actual) != bool(code & 128)) return 1;
        } else {
            float expected = exponent == 0 ? std::ldexp(float(mantissa), -9) : std::ldexp(float(8 + mantissa), exponent - 10);
            if (code & 128) expected = -expected;
            uint32_t want;
            std::memcpy(&want, &expected, sizeof(want));
            if (bits != want) {
                std::fprintf(stderr, "E4M3FN code %u decoded incorrectly\n", code);
                return 1;
            }
        }
    }
    std::puts("PASS FP8 decode: all 256 codes including subnormals, signed zeros and NaNs");
}
