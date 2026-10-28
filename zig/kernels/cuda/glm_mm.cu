// GLM dense linears, ours: one tensor-core kernel for any row count, 64-groups then K slices in order; row-stable bits.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "glm_qmm_frag.cuh"
#include "glm_q4_pair.cuh"

namespace {

using namespace qmm_frag;

// 16 codes from a packed chunk (ops.cu's layout), stored + 2^(BITS-1)
template <int BITS>
__device__ __forceinline__ void codes16(const uint32_t* __restrict__ p, int (&q)[16]) {
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 8] >> (4 * (i % 8))) & 15u) - 8;
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) q[i] = (int)((p[i / 4] >> (8 * (i % 4))) & 255u) - 128;
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) q[i] = (int)((p[i / 5] >> (6 * (i % 5))) & 63u) - 32;
        q[15] = (int)((p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4)) - 32;
    }
}

template <int BITS>
__device__ __forceinline__ void pack16(const int (&q)[16], uint32_t* __restrict__ out) {
    constexpr int half = 1 << (BITS - 1);
    uint32_t w[BITS / 2];
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) w[i] = 0u;
    if constexpr (BITS == 4) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 8] |= (uint32_t)(q[i] + half) << (4 * (i % 8));
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i / 4] |= (uint32_t)(q[i] + half) << (8 * (i % 4));
    } else {
#pragma unroll
        for (int i = 0; i < 15; ++i) w[i / 5] |= (uint32_t)(q[i] + half) << (6 * (i % 5));
        const uint32_t last = (uint32_t)(q[15] + half);
#pragma unroll
        for (int j = 0; j < 3; ++j) w[j] |= ((last >> (2 * j)) & 3u) << 30;
    }
#pragma unroll
    for (int i = 0; i < BITS / 2; ++i) out[i] = w[i];
}

// input of value v (0..15) of lane ``lane``'s chunk within its group of 64
__device__ __forceinline__ int frag_k(int lane, int v) {
    const int t = lane & 3, kt = v >> 2, j = v & 3;
    return kt * 16 + 2 * t + (j & 1) + (j >> 1) * 8;
}

// the B-fragment bf16 pairs of a chunk: pairs 2 kt and 2 kt + 1 are step kt's b0 and b1
template <int BITS, bool PAIR = false, bool AFFINE = false>
__device__ __forceinline__ void bpairs(const uint32_t* __restrict__ p, uint32_t (&b)[8]) {
    if constexpr (BITS == 16) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[i] = p[i];
    } else if constexpr (AFFINE) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t packed = tf_glm_q4::biased_pair(p[i / 4], i % 4);
#if __CUDA_ARCH__ >= 900
            asm("sub.rn.bf16x2 %0, %1, %2;" : "=r"(b[i]) : "r"(packed), "r"(0x43004300u));
#else
            asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(b[i]) : "r"(packed), "r"(0x3f803f80u), "r"(0xc300c300u));
#endif
        }
    } else if constexpr (BITS == 4 && PAIR) {
#pragma unroll
        for (int i = 0; i < 8; ++i) b[i] = tf_glm_q4::decode_pair(p[i / 4], i % 4);
    } else {
        int q[16];
        codes16<BITS>(p, q);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __nv_bfloat162 h = __floats2bfloat162_rn((float)q[2 * i], (float)q[2 * i + 1]);   // exact integers
            b[i] = *reinterpret_cast<const uint32_t*>(&h);
        }
    }
}

template <int BITS, int BM, int WM, int STAGES, bool AFFINE = false, int COLWARPS = 4, int COLS = COLWARPS * 16>
struct Tile {
    static constexpr int WN = COLWARPS, BN = COLS, NT = BN / WN / 8, MT = BM / WM / 16, THREADS = WM * WN * 32;
    static constexpr int WPC = BITS / 2;
    static constexpr int ROW = 128, CHUNKS = 8;               // a row's 64 inputs: 128 bytes, 8 chunks of 16
    static constexpr int X = BM * ROW;
    static constexpr int W = (BN / 8) * 32 * WPC * 4;                // 8 column tiles' chunks
    static constexpr int S = BN * 2;                       // their scales (fp16)
    static constexpr int STAGE = X + W + S + (AFFINE ? S + BM * 4 : 0);
    static constexpr int SMEM = STAGES * STAGE;
};

// ROWS_FIRST: row blocks fastest, so the blocks reading one weight tile run together and share it in L2 (prompts)
template <int BITS, int BM, int WM, int STAGES, bool ROWS_FIRST = false, bool PAIR = false, bool AFFINE = false, bool XS = false, bool RECON = false, int COLWARPS = 4, int COLS = COLWARPS * 16>
__device__ __forceinline__ void mm_kernel(const __nv_bfloat16* __restrict__ x, long long ldx,
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
        int f32, int M, int N, int K, int SK, const float* xs = nullptr) {
    using T = Tile<BITS, BM, WM, STAGES, AFFINE, COLWARPS, COLS>;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int KG = K >> 6, per = KG / SK, N8 = N >> 3;
    const int n0 = (ROWS_FIRST ? blockIdx.y : blockIdx.x) * T::BN, m0 = (ROWS_FIRST ? blockIdx.x : blockIdx.y) * BM;
    const int slice = blockIdx.z, g0 = slice * per;
    const int t0 = n0 >> 3;                                   // the block's first column tile
    auto stage = [&](int st) { return buf + st * T::STAGE; };
    auto loadX = [&](int st, int g) {
        unsigned char* p = stage(st);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + (long long)row * ldx + g * 64 + ch * 8, m0 + r < M);
        }
    };
    auto loadW = [&](int st, int g) {
        unsigned char* pw = stage(st) + T::X;
        constexpr int PER_TILE = 32 * T::WPC * 4 / 16;        // 16-byte pieces of one column tile's chunks
        for (int c = tid; c < (T::BN / 8) * PER_TILE; c += T::THREADS) {
            const int tt = c / PER_TILE, off = c % PER_TILE;
            if (t0 + tt >= N8) continue;
            const long long src = ((long long)(t0 + tt) * KG + g) * 32 * T::WPC * 4 + off * 16;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + src);
        }
        if constexpr (BITS != 16) {
            unsigned char* ps = pw + T::W;
            for (int tt = tid; tt < T::BN / 8; tt += T::THREADS) {
                if (t0 + tt >= N8) continue;
                const long long table = ((long long)(t0 + tt) * KG + g) * 8;
                cp16(ps + tt * 16, s + table);
                if constexpr (AFFINE) cp16(ps + T::S + tt * 16, s + (long long)N * KG + table);
            }
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
    // dependent launch: first stages' weights stream in early; inputs and outputs only after griddepcontrol.wait
    grid_launch();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st)
        if (st < per) loadW(st, g0 + st);
    grid_wait();
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < per) loadX(st, g0 + st);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) {
            loadX(next % STAGES, g0 + next);
            loadW(next % STAGES, g0 + next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        if constexpr (AFFINE && !XS && !RECON) {
            float* sums = reinterpret_cast<float*>(stage(it % STAGES) + T::X + T::W + 2 * T::S);
            for (int r = tid / 8; r < BM; r += T::THREADS / 8) {
                const int ch = tid & 7;
                const auto* vals = reinterpret_cast<const __nv_bfloat16*>(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
                float v = __bfloat162float(vals[0]) + __bfloat162float(vals[1]);
#pragma unroll
                for (int j = 2; j < 8; ++j) v = __bfloat162float(vals[j]) + v;
#pragma unroll
                for (int o = 4; o > 0; o >>= 1) v = v + __shfl_xor_sync(0xffffffffu, v, o);
                if (ch == 0) sums[r] = v;
            }
            __syncthreads();
        }
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __half* ps = reinterpret_cast<const __half*>(p + T::X + T::W);
        uint32_t b[T::NT][8];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            bpairs<BITS, PAIR, AFFINE>(pw + ((wn * T::NT + j) * 32 + lane) * T::WPC, b[j]);
            if constexpr (RECON) {
                const auto* table = reinterpret_cast<const __nv_bfloat16*>(ps);
                const int ci = (wn * T::NT + j) * 8 + (lane >> 2);
                const uint32_t scale = (uint32_t)__bfloat16_as_ushort(table[ci]) * 0x10001u;
                const uint32_t bias = (uint32_t)__bfloat16_as_ushort(table[T::BN + ci]) * 0x10001u;
#pragma unroll
                for (int t = 0; t < 8; ++t)
                    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(b[j][t]) : "r"(b[j][t]), "r"(scale), "r"(bias));
            }
        }
        float d[T::MT][T::NT][4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (RECON) mma(acc[i][j], a[i], b[j][2 * kt], b[j][2 * kt + 1]);
                    else if (kt == 0) mma0(d[i][j], a[i], b[j][0], b[j][1]);
                    else mma(d[i][j], a[i], b[j][2 * kt], b[j][2 * kt + 1]);
                }
        }
#pragma unroll
        for (int j = 0; j < T::NT && !RECON; ++j) {
            float s0 = 1.0f, s1 = 1.0f;
            float b0 = 0.0f, b1 = 0.0f;
            const int ci = (wn * T::NT + j) * 8 + (lane & 3) * 2;
            if constexpr (AFFINE) {
                const auto* tables = reinterpret_cast<const __nv_bfloat16*>(ps);
                s0 = __bfloat162float(tables[ci]); s1 = __bfloat162float(tables[ci + 1]);
                b0 = __bfloat162float(tables[T::BN + ci]); b1 = __bfloat162float(tables[T::BN + ci + 1]);
            } else if constexpr (BITS != 16) {
                const __half2 h = *reinterpret_cast<const __half2*>(ps + ci);
                s0 = __low2float(h);
                s1 = __high2float(h);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[i][j][e] = __fmaf_rn(d[i][j][e], (e & 1) ? s1 : s0, acc[i][j][e]);
                    if constexpr (AFFINE) {
                        const float* sums = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
                        const int ri = wm * (BM / WM) + i * 16 + (lane >> 2) + (e >> 1) * 8;
                        const float sum = XS ? ((m0 + ri < M) ? xs[(long long)(m0 + ri) * KG + g0 + it] : 0.0f) : sums[ri];
                        acc[i][j][e] = __fmaf_rn(sum, (e & 1) ? b1 : b0, acc[i][j][e]);
                    }
                }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + (wn * T::NT + j) * 8 + (lane & 3) * 2;
            if (col >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (SK > 1) {
                    float* dst = part + ((long long)slice * M + row) * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else if (f32) {
                    float* dst = reinterpret_cast<float*>(out) + (long long)row * N + col;
                    dst[0] = v0;
                    dst[1] = v1;
                } else {
                    *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + (long long)row * N + col) =
                        __floats2bfloat162_rn(v0, v1);
                }
            }
        }
}

}  // namespace

// The K slices' partials added in slice order, then rounded once.
extern "C" __global__ void glm_mm_reduce(const float* __restrict__ part, void* __restrict__ out, long long total, int SK, int f32) {
    qmm_frag::grid_launch();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = part[i];
    for (int sl = 1; sl < SK; ++sl) acc = acc + part[sl * total + i];
    if (f32) reinterpret_cast<float*>(out)[i] = acc;
    else reinterpret_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16_rn(acc);
}

// one instance a (bits, row bucket): (16, 1 warp row), (32, 1), (64, 2), (128, 2); three stages
#define GLM_MM_ALL(B)                                                                                                  \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_16(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 16, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(128) glm_mm##B##_32(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 32, 1, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_64(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 64, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }     \
    extern "C" __global__ void __launch_bounds__(256) glm_mm##B##_128(const __nv_bfloat16* __restrict__ x, long long ldx, \
        const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, \
        int f32, int M, int N, int K, int SK) { mm_kernel<B, 128, 2, 3>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
// prompts: 128-row blocks, row blocks fastest (one slice: a prompt's rows fill the GPU)
extern "C" __global__ void __launch_bounds__(256) glm_mm16_rows(const __nv_bfloat16* __restrict__ x, long long ldx,
    const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
    int f32, int M, int N, int K, int SK) { mm_kernel<16, 128, 2, 3, true>(x, ldx, w, s, out, part, f32, M, N, K, SK); }
GLM_MM_ALL(4)
GLM_MM_ALL(6)
GLM_MM_ALL(8)
GLM_MM_ALL(16)

// Experimental packed bf16 decode; the tiling, MMA chains and scale/reduce arithmetic stay the same.
extern "C" __global__ void __launch_bounds__(128) glm_mm4_pair_16(const __nv_bfloat16* __restrict__ x, long long ldx,
    const uint32_t* __restrict__ w, const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part,
    int f32, int M, int N, int K, int SK) {
    mm_kernel<4, 16, 1, 3, false, true>(x, ldx, w, s, out, part, f32, M, N, K, SK);
}

// Affine groups retain separate BF16 scale/bias and the ordered FP32 dot and input-sum terms.
#define GLM_AFFINE(BM, WM, THREADS) \
extern "C" __global__ void __launch_bounds__(THREADS) glm_mm_affine4_##BM( \
    const __nv_bfloat16* x, long long ldx, const uint32_t* w, const __half* s, void* out, float* part, \
    int f32, int M, int N, int K, int SK) { \
    mm_kernel<4, BM, WM, 3, false, false, true>(x, ldx, w, s, out, part, f32, M, N, K, SK); \
}
GLM_AFFINE(16, 1, 128)
GLM_AFFINE(32, 1, 128)
GLM_AFFINE(64, 2, 256)
GLM_AFFINE(128, 2, 256)
#undef GLM_AFFINE

// Native affine input sums computed once per row/group, preserving the existing eight-lane reduction order.
extern "C" __global__ void __launch_bounds__(128) glm_affine_sums(const __nv_bfloat16* x, long long ldx,
    float* xs, int M, int KG) {
    qmm_frag::grid_wait();
    const int r = blockIdx.x, g = blockIdx.y * 16 + threadIdx.x / 8, ch = threadIdx.x & 7;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = g < KG ? __bfloat162float(x[(long long)r * ldx + g * 64 + ch * 8 + j]) : 0.0f;
    float sum = v[0] + v[1];
#pragma unroll
    for (int j = 2; j < 8; ++j) sum = v[j] + sum;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum = sum + __shfl_xor_sync(0xffffffffu, sum, o);
    if (ch == 0 && g < KG && r < M) xs[(long long)r * KG + g] = sum;
}

#define GLM_AFFINE_ROUTE(NAME, BM, WM, ST, CW, BN, RECON) \
extern "C" __global__ void __launch_bounds__(WM * CW * 32) NAME( \
    const __nv_bfloat16* x, long long ldx, const uint32_t* w, const __half* s, void* out, float* part, \
    int f32, int M, int N, int K, int SK, const float* xs) { \
    mm_kernel<4, BM, WM, ST, false, false, true, true, RECON, CW, BN>(x, ldx, w, s, out, part, f32, M, N, K, SK, xs); \
}
GLM_AFFINE_ROUTE(glm_affine_xs16, 16, 1, 3, 4, 64, false)
GLM_AFFINE_ROUTE(glm_affine_xs32, 32, 1, 3, 4, 64, false)
GLM_AFFINE_ROUTE(glm_affine_xs64, 64, 2, 3, 4, 64, false)
GLM_AFFINE_ROUTE(glm_affine_xs128, 128, 2, 3, 4, 64, false)
GLM_AFFINE_ROUTE(glm_affine_wide16, 16, 1, 4, 8, 128, false)
GLM_AFFINE_ROUTE(glm_affine_wide16s6, 16, 1, 6, 4, 128, false)
GLM_AFFINE_ROUTE(glm_affine_prefill64, 64, 1, 4, 4, 64, true)
GLM_AFFINE_ROUTE(glm_affine_prefill128, 128, 2, 2, 2, 128, true)
#undef GLM_AFFINE_ROUTE

// -- tiling at load: ops.cu's packed rows and fp16 scales [N][K / 64] -> the tiled chunks and scales ---------------
template <int BITS>
__device__ __forceinline__ void tile_packed(const uint32_t* __restrict__ src, const __half* __restrict__ ss,
                                            uint32_t* __restrict__ dst, __half* __restrict__ ds, int N, int K) {
    constexpr int WPC = BITS / 2;
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;   // a destination chunk: (tile, group, lane)
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
    const uint32_t* row = src + (long long)n * (K >> 4) * WPC;
    int q[16];
#pragma unroll
    for (int v = 0; v < 16; ++v) {
        const int k = g * 64 + frag_k(lane, v);
        int sq[16];
        codes16<BITS>(row + (long long)(k >> 4) * WPC, sq);
        q[v] = sq[k & 15];
    }
    pack16<BITS>(q, dst + c * WPC);
    if (lane < 8) ds[(tg * 8) + lane] = ss[(long long)(tt * 8 + lane) * KG + g];
}

extern "C" __global__ void glm_mm_tile4(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<4>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile6(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<6>(src, ss, dst, ds, N, K); }
extern "C" __global__ void glm_mm_tile8(const uint32_t* src, const __half* ss, uint32_t* dst, __half* ds, int N, int K) { tile_packed<8>(src, ss, dst, ds, N, K); }

// Scale and bias are consecutive BF16 tables; tile_packed only moves their raw 16-bit storage.
extern "C" __global__ void glm_mm_tile_affine4(const uint32_t* src, const __half* ss,
                                             uint32_t* dst, __half* ds, int N, int K) {
    tile_packed<4>(src, ss, dst, ds, N, K);
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int KG = K / 64;
    if (c >= (long long)(N / 8) * KG * 32 || (c & 31) >= 8) return;
    const long long tg = c / 32;
    const int lane = c & 31, g = tg % KG, tt = tg / KG;
    ds[(long long)N * KG + tg * 8 + lane] = ss[(long long)N * KG + (tt * 8 + lane) * KG + g];
}

// bf16 rows [N][K] -> the tiled chunks (16 bf16 a chunk, in pair order)
extern "C" __global__ void glm_mm_tile16(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int N, int K) {
    const int KG = K >> 6, N8 = N >> 3;
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (long long)N8 * KG * 32) return;
    const int lane = (int)(c & 31);
    const long long tg = c >> 5;
    const int g = (int)(tg % KG), tt = (int)(tg / KG);
    const int n = tt * 8 + (lane >> 2);
#pragma unroll
    for (int v = 0; v < 16; ++v) dst[c * 16 + v] = src[(long long)n * K + g * 64 + frag_k(lane, v)];
}

#include "glm_fp8_mm.cuh"
