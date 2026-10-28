// Native scalar affine Q4 arithmetic, parallel across groups and candidates; no PyTorch reduction reassociation.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ float wide(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t narrow(float x) {
    const uint32_t u = __float_as_uint(x);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint32_t code(float x, float s, float b) {
    const float q = __fdiv_rn(__fsub_rn(x, b), s);
    return __float2uint_rn(fmaxf(0.0f, fminf(15.0f, q)));
}

// One warp per group, sixteen candidate lanes, serial FP32 losses, strict first minimum and low-first nibbles.
extern "C" __global__ __launch_bounds__(128) void tf_glm_affine_quant(
    const uint16_t* src, uint32_t* words, uint16_t* scales, uint16_t* biases,
    uint32_t* status, uint64_t groups) {
    const uint64_t group = uint64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    if (group >= groups) return;
    const int lane = threadIdx.x & 31;
    float lo = CUDART_INF_F, hi = -CUDART_INF_F;
    bool finite = true;
    #pragma unroll 1
    for (int j = 0; j < 64; ++j) {
        const float x = wide(src[group * 64 + j]);
        finite &= isfinite(x);
        lo = x < lo ? x : lo;
        hi = x > hi ? x : hi;
    }
    if (!finite) { if (lane == 0) atomicOr(status, 1u); return; }
    float loss = CUDART_INF_F;
    uint16_t scale = 0, bias = 0;
    bool valid = true;
    if (lane < 16) {
        // Explicit FP64 operations reproduce the scalar host alpha before its one FP32 conversion.
        const float alpha = __double2float_rn(__dsub_rn(1.0, __ddiv_rn(__dmul_rn(0.45, double(lane)), 15.0)));
        const float low = __fmul_rn(lo, alpha), high = __fmul_rn(hi, alpha);
        scale = narrow(fmaxf(__fdiv_rn(__fsub_rn(high, low), 15.0f), 1e-8f));
        bias = narrow(low);
        const float s = wide(scale), b = wide(bias);
        valid = isfinite(s) && isfinite(b) && s > 0;
        if (valid) {
            loss = 0;
            #pragma unroll 1
            for (int j = 0; j < 64; ++j) {
                const float x = wide(src[group * 64 + j]);
                const float q = float(code(x, s, b));
                const float r = __fsub_rn(__fadd_rn(__fmul_rn(q, s), b), x);
                loss = __fadd_rn(loss, __fmul_rn(r, r));
            }
            valid = isfinite(loss);
        }
    }
    if (__any_sync(0xffffffffu, !valid)) { if (lane == 0) atomicOr(status, 2u); return; }
    float best = __shfl_sync(0xffffffffu, loss, 0);
    int winner = 0;
    #pragma unroll
    for (int c = 1; c < 16; ++c) {
        const float trial = __shfl_sync(0xffffffffu, loss, c);
        if (trial < best) { best = trial; winner = c; }
    }
    scale = uint16_t(__shfl_sync(0xffffffffu, uint32_t(scale), winner));
    bias = uint16_t(__shfl_sync(0xffffffffu, uint32_t(bias), winner));
    if (lane == 0) { scales[group] = scale; biases[group] = bias; }
    if (lane < 8) {
        uint32_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            packed |= code(wide(src[group * 64 + lane * 8 + j]), wide(scale), wide(bias)) << (4 * j);
        words[group * 8 + lane] = packed;
    }
}
