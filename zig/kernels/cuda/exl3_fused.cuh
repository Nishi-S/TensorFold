// Optional EXL3 decode epilogues: preserve the grouped projection's warp/split sums and existing rounding points.
#pragma once

namespace tf_exl3 {

struct FusionArgs {
    const half* sv0;
    const half* sv1;
    const half* su2;
    void* out;
    int* done;
    float limit;
    int act_mode;
    int load_mode;
};

__device__ __forceinline__ float fused_bf16(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

// Every block publishes its partials before the last block reads them; it never waits for another resident block.
__device__ __forceinline__ bool fusion_last(int* counter, int parts) {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1) == parts - 1;
    __syncthreads();
    return last != 0;
}

// `reduced` is [16,128]; gate/up uses ordered global partials, down uses its single split's reduced shared values.
template <int FUSE>
__device__ __forceinline__ void fusion_finish(
    const FusionArgs& f, float* z, const int* members, const float* reduced,
    int e, int item, int block, int p, int n, int sk, int mtile, int member_tiles) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int* counter = nullptr;
    if constexpr (FUSE == 2) {
        counter = f.done + (item * member_tiles + mtile) * (n / 128) + block;
        if (!fusion_last(counter, 2 * sk)) return;
    }
    const int col = block * 128 + lane * 4;
    for (int row = warp; row < 16; row += 4) {
        const int pair = members[row];
        if (pair < 0) continue;
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sg = 0.f, su = 0.f;
            if constexpr (FUSE == 1) {
                sg += reduced[row * 128 + lane * 4 + j];
            } else {
                const volatile float* partial = z;
                for (int split = 0; split < sk; ++split) {
                    sg += partial[((size_t)split * p + pair) * n + col + j];
                    su += partial[((size_t)(sk + split) * p + pair) * n + col + j];
                }
            }
            gv[j] = sg;
            uv[j] = su;
        }
        fwht128(gv, lane);
        if constexpr (FUSE == 1) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                static_cast<float*>(f.out)[(size_t)pair * n + col + j] =
                    gv[j] * HAD_SCALE * __half2float(f.sv0[(size_t)e * n + col + j]);
        } else {
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = gv[j] * HAD_SCALE * __half2float(f.sv0[(size_t)e * n + col + j]);
                float uu = uv[j] * HAD_SCALE * __half2float(f.sv1[(size_t)e * n + col + j]);
                if (f.act_mode == 0) { gg = fused_bf16(gg); uu = fused_bf16(uu); }
                if (f.limit > 0) {
                    gg = gg > f.limit ? f.limit : gg;
                    uu = uu > f.limit ? f.limit : (uu < -f.limit ? -f.limit : uu);
                }
                float act = gg / (1.f + expf(-gg));
                if (f.act_mode == 0) act = fused_bf16(act);
                act *= uu;
                if (f.act_mode == 0) act = fused_bf16(act);
                v[j] = act * __half2float(f.su2[(size_t)e * n + col + j]);
            }
            fwht128(v, lane);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                static_cast<half*>(f.out)[(size_t)pair * n + col + j] = __float2half_rn(v[j] * HAD_SCALE);
        }
    }
    if constexpr (FUSE == 2) {
        __syncthreads();
        if (threadIdx.x == 0) atomicExch(counter, 0);
    }
}

}  // namespace tf_exl3
