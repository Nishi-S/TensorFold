// Optional one-step trellis prefetch for the native mcg/4-bit grouped kernel.
#pragma once

#include "exl3_decode.cuh"

namespace tf_exl3 {

// Volatile loads stay ahead of the current tile's volatile MMA instructions; weights are immutable during a run.
__device__ __forceinline__ void prefetch_words4(const uint32_t* tile, int lane, uint32_t (&w)[lane_words<8>()]) {
    int word, offset;
    lane_start<8>(lane, word, offset);
#pragma unroll
    for (int i = 0; i < lane_words<8>(); ++i) {
        const uint32_t* p = tile + (word + i) % tile_words<8>();
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(w[i]) : "l"(p) : "memory");
    }
}

// Every lane keeps the next step's words in registers; the last step never reads beyond its warp's K slice.
template <int NT>
__device__ __forceinline__ void warp_tiles_pf1(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0,
        const half* x0, const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4]) {
    constexpr int TW = tile_words<8>(), LW = lane_words<8>();
    if (nkt <= 0) return;
    const size_t stride = (size_t)ntiles * TW;
    const uint32_t* tile = T + (size_t)kt0 * stride + (size_t)nt0 * TW;
    uint32_t next[NT][LW];
#pragma unroll
    for (int j = 0; j < NT; ++j) prefetch_words4(tile + j * TW, lane, next[j]);
    for (int i = 0; i < nkt; ++i) {
        uint32_t words[NT][LW];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < LW; ++v) words[j][v] = next[j][v];
        if (i + 1 < nkt) {
            tile += stride;
#pragma unroll
            for (int j = 0; j < NT; ++j) prefetch_words4(tile + j * TW, lane, next[j]);
        }
        const int k = (kt0 + i) * 16;
        const uint32_t a[4] = {ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
                               ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
                               ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            uint32_t b0[2], b1[2];
            decode_lane<8, 1>(words[j], lane, b0, b1);
            mma16816(acc[j][0], a, b0);
            mma16816(acc[j][1], a, b1);
        }
    }
}

// One aligned vector per lane covers four complete native 128-byte tiles across the warp.
__device__ __forceinline__ uint4 prefetch_vector(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

// A bounded ring of native trellis steps; stage belongs to this warp and becomes its reduction buffer afterwards.
template <int NT, int PD>
__device__ __forceinline__ void warp_tiles_nc(const uint32_t* T, int ntiles, int kt0, int nkt, int nt0,
        const half* x0, const half* x1, bool ok0, bool ok1, int lane, float (&acc)[NT][2][4], uint32_t* stage) {
    static_assert(NT % 4 == 0 && (PD == 1 || PD == 2 || PD == 4), "vector ring geometry");
    constexpr int TW = tile_words<8>(), NV = NT / 4;
    const size_t stride = (size_t)ntiles * TW;
    const uint32_t* base = T + (size_t)kt0 * stride + (size_t)nt0 * TW;
    uint4 ring[PD][NV];
    uint32_t aring[PD][4];
#pragma unroll
    for (int d = 0; d < PD; ++d) if (d < nkt) {
#pragma unroll
        for (int v = 0; v < NV; ++v) ring[d][v] = prefetch_vector(base + d * stride + v * 128 + lane * 4);
        const int k = (kt0 + d) * 16;
        aring[d][0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
        aring[d][1] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
        aring[d][2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
        aring[d][3] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
    }
    for (int block = 0; block < nkt; block += PD) {
#pragma unroll
        for (int d = 0; d < PD; ++d) if (block + d < nkt) {
            const int step = block + d;
            uint32_t a[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) a[v] = aring[d][v];
            __syncwarp();
#pragma unroll
            for (int v = 0; v < NV; ++v)
                *reinterpret_cast<uint4*>(stage + v * 128 + lane * 4) = ring[d][v];
            if (step + PD < nkt) {
#pragma unroll
                for (int v = 0; v < NV; ++v)
                    ring[d][v] = prefetch_vector(base + (size_t)(step + PD) * stride + v * 128 + lane * 4);
                const int k = (kt0 + step + PD) * 16;
                aring[d][0] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u;
                aring[d][1] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u;
                aring[d][2] = ok0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u;
                aring[d][3] = ok1 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u;
            }
            __syncwarp();
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t words[lane_words<8>()], b0[2], b1[2];
                load_lane_words<8>(stage + j * TW, lane, words);
                decode_lane<8, 1>(words, lane, b0, b1);
                mma16816(acc[j][0], a, b0);
                mma16816(acc[j][1], a, b1);
            }
        }
    }
    __syncwarp();
}

}  // namespace tf_exl3
