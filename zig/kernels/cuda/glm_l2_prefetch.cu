// Optional bounded cache hints over immutable GLM weights; no data or scratch is written.
#include <stdint.h>
#include "glm_l2_range.cuh"

__device__ __forceinline__ void lines(const unsigned char* ptr, uint64_t bytes) {
    const uint64_t first = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 128;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x * 128;
    for (uint64_t offset = first; offset < bytes; offset += step)
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + offset) : "memory");
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_l2_prefetch(const unsigned char* ptr, uint64_t bytes) {
    lines(ptr, bytes);
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_l2_prefetch_bulk(const unsigned char* ptr, uint64_t bytes) {
#if __CUDA_ARCH__ >= 900
    const auto range = tf_glm_l2::aligned((uint64_t)ptr, bytes, 16);
    const uint64_t first = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 32768;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x * 32768;
    for (uint64_t offset = first; offset < range.bytes; offset += step) {
        const uint64_t remaining = range.bytes - offset;
        const uint32_t count = (uint32_t)(remaining < 32768 ? remaining : 32768);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                     :: "l"(ptr + range.start + offset), "r"(count) : "memory");
    }
#else
    lines(ptr, bytes);
#endif
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_l2_prefetch_touch(const unsigned char* ptr, uint64_t bytes) {
    const auto range = tf_glm_l2::aligned((uint64_t)ptr, bytes, 4);
    const uint64_t first = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * 128;
    const uint64_t step = (uint64_t)gridDim.x * blockDim.x * 128;
    for (uint64_t offset = first; offset < range.bytes; offset += step) {
        uint32_t word;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(word) : "l"(ptr + range.start + offset) : "memory");
    }
}
