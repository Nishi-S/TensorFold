// Weight-only E4M3FN: BF16 activations, FP32 dots per 64 K values and scales shared by two adjacent dots.
#pragma once
#include "glm_qmm_frag.cuh"
#include "glm_fp8_decode.cuh"

namespace tf_glm_fp8 {
using namespace qmm_frag;

template <int BM, int WM>
__device__ __forceinline__ void matmul(const __nv_bfloat16* __restrict__ x, long long ldx,
        const uint8_t* __restrict__ weights, const float* __restrict__ scales, void* __restrict__ out,
        float* __restrict__ part, int wide, int M, int N, int K, int SK) {
    constexpr int WN = 4, NT = 2, MT = BM / WM / 16, THREADS = WM * WN * 32;
    constexpr int X = BM * 256, W = 64 * 128, S = 64 * 4, STAGE = X + W + S;
    extern __shared__ __align__(128) unsigned char memory[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN, n0 = blockIdx.x * 64, m0 = blockIdx.y * BM;
    const int groups = K / 128, per = groups / SK, slice = blockIdx.z, first = slice * per;
    auto stage = [&](int i) { return memory + (i & 1) * STAGE; };
    auto loadWeights = [&](int i, int group) {
        unsigned char* p = stage(i) + X;
        for (int c = tid; c < 64 * 8; c += THREADS) {
            const int col = c / 8, piece = c % 8;
            const int safe_col = min(n0 + col, N - 1);
            cp16z(p + col * 128 + piece * 16,
                weights + (long long)safe_col * K + group * 128 + piece * 16, n0 + col < N);
        }
        float* s = reinterpret_cast<float*>(p + W);
        for (int col = tid; col < 64; col += THREADS)
            s[col] = n0 + col < N ? scales[(long long)(n0 + col) * groups + group] : 0.0f;
    };
    auto loadInputs = [&](int i, int group) {
        unsigned char* p = stage(i);
        for (int c = tid; c < BM * 16; c += THREADS) {
            const int r = c / 16, ch = c % 16, safe_row = min(m0 + r, M - 1);
            cp16z(p + r * 256 + swz<16>(r, ch) * 16,
                x + (long long)safe_row * ldx + group * 128 + ch * 8, m0 + r < M);
        }
    };
    float acc[MT][NT][4] = {};
    grid_launch();
    loadWeights(0, first);
    grid_wait();
    loadInputs(0, first);
    commit();
    for (int g = 0; g < per; ++g) {
        wait<0>();
        __syncthreads();
        if (g + 1 < per) {
            loadWeights(g + 1, first + g + 1);
            loadInputs(g + 1, first + g + 1);
        }
        commit();
        const unsigned char* p = stage(g);
        const uint8_t* w = p + X;
        const float* s = reinterpret_cast<const float*>(w + W);
        float dot[MT][NT][4];
#pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * 256 + swz<16>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint8_t* q = w + ((wn * NT + j) * 8 + (lane >> 2)) * 128 + kt * 16 + (lane & 3) * 2;
                const uint32_t b0 = (uint32_t)bf16(q[0]) | ((uint32_t)bf16(q[1]) << 16);
                const uint32_t b1 = (uint32_t)bf16(q[8]) | ((uint32_t)bf16(q[9]) << 16);
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if ((kt & 3) == 0) mma0(dot[i][j], a[i], b0, b1);
                    else mma(dot[i][j], a[i], b0, b1);
                }
            }
            if ((kt & 3) == 3) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int col = (wn * NT + j) * 8 + (lane & 3) * 2;
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int e = 0; e < 4; ++e)
                            acc[i][j][e] = __fmaf_rn(dot[i][j][e], s[col + (e & 1)], acc[i][j][e]);
                }
            }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + (wn * NT + j) * 8 + (lane & 3) * 2;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float a = acc[i][j][h * 2], b = acc[i][j][h * 2 + 1];
                if (SK > 1) {
                    float* dst = part + ((long long)slice * M + row) * N + col;
                    dst[0] = a;
                    dst[1] = b;
                } else if (wide) {
                    float* dst = reinterpret_cast<float*>(out) + (long long)row * N + col;
                    dst[0] = a;
                    dst[1] = b;
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)row * N + col) =
                        __floats2bfloat162_rn(a, b);
                }
            }
        }
}
} // namespace tf_glm_fp8

#define GLM_FP8(BM, WM, THREADS) \
extern "C" __global__ void __launch_bounds__(THREADS) glm_fp8_##BM(const __nv_bfloat16* x, long long ldx, \
    const uint8_t* w, const float* s, void* out, float* part, int wide, int M, int N, int K, int SK) { \
    tf_glm_fp8::matmul<BM, WM>(x, ldx, w, s, out, part, wide, M, N, K, SK); \
}
GLM_FP8(16, 1, 128)
GLM_FP8(32, 1, 128)
GLM_FP8(64, 2, 256)
GLM_FP8(128, 2, 256)
#undef GLM_FP8
