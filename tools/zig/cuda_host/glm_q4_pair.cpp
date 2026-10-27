// Exhaust the actual nibble-placement helper, independently decoding its bf16 bits before the exact subtraction.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include "../../../zig/kernels/cuda/glm_q4_pair.cuh"

static float bf16(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

int main() {
    unsigned checks = 0;
    for (unsigned pair = 0; pair < 4; ++pair) {
        for (unsigned byte = 0; byte < 256; ++byte) {
            for (uint32_t background : {0u, 0xffffffffu, 0x12345678u, 0x87654321u}) {
                const uint32_t mask = 255u << (8 * pair);
                const uint32_t word = (background & ~mask) | (byte << (8 * pair));
                const uint32_t got = tf_glm_q4::biased_pair(word, pair);
                if (bf16(uint16_t(got)) - 136.0f != int(byte & 15) - 8 ||
                    bf16(uint16_t(got >> 16)) - 136.0f != int(byte >> 4) - 8) return 1;
                ++checks;
            }
        }
    }
    std::printf("PASS Q4 pair packing: %u placements; CUDA instruction and GEMM equivalence need device tests\n", checks);
}
