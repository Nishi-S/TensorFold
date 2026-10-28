// Exhaustive short ranges and chunk edges prove every bulk/touch address stays inside its byte range.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include "../../../zig/kernels/cuda/glm_l2_range.cuh"

static void check(uint64_t ptr, uint64_t bytes, uint64_t alignment, uint64_t stride) {
    const auto w = tf_glm_l2::aligned(ptr, bytes, alignment);
    assert(w.bytes == 0 || (w.start < bytes && w.bytes <= bytes - w.start));
    assert(((ptr + w.start) % alignment) == 0 && w.bytes % alignment == 0);
    for (uint64_t offset = 0; offset < w.bytes; offset += stride) {
        const auto left = w.bytes - offset;
        const auto n = alignment == 4 ? 4 : (left < stride ? left : stride);
        assert(n > 0 && n <= stride && n % alignment == 0);
        assert(w.start + offset + n <= bytes);
    }
}
int main() {
    unsigned cases = 0;
    for (uint64_t offset = 0; offset < 128; ++offset) {
        for (uint64_t bytes = 0; bytes <= 513; ++bytes) {
            check(4096 + offset, bytes, 16, 32768);
            check(4096 + offset, bytes, 4, 128);
            ++cases;
        }
        for (const uint64_t bytes : {32767, 32768, 32769, 65535, 65536, 65537}) {
            check(4096 + offset, bytes, 16, 32768);
            check(4096 + offset, bytes, 4, 128);
            ++cases;
        }
    }
    std::printf("PASS glm_l2_range: %u byte ranges, bulk and touch alignment/bounds\n", cases);
}
