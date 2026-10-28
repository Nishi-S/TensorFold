// Native HC schedules inspired by Mia patches 0004/0019; see LICENSES/MiaAI-Lab-MIT.txt.
#pragma once
#include <mma.h>
#include "glm_triton.cuh"

namespace glm_hc_fast {
// Preserve the legacy 128-lane square sum using one warp and its four logical lane groups.
__device__ __forceinline__ float squares(const __nv_bfloat16* x, int lane) {
  float sums[4];
  for (int j = 0; j < 4; ++j) {
    float v = 0;
    for (int k = 0; k < 8; ++k) {
      float a = __bfloat162float(x[k * 128 + j * 32 + lane]);
      v = __fmaf_rn(a, a, v);
    }
    for (int o = 16; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
    sums[j] = v;
  }
  return __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
}
}

// Three independent eight-dot partitions reuse weights across four rows and preserve every legacy rounding.
extern "C" __global__ void __launch_bounds__(288) tf_glm_hc_partial_dec(
    const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int b = blockIdx.y, m = blockIdx.z * 8 + warp;
  const int64_t first = (int64_t)blockIdx.x * 4;
  if (warp < 8) {
    const int half = lane / 16, col = lane % 16;
    float weights[4][8];
    for (int step = 0; step < 4; ++step)
      for (int j = 0; j < 8; ++j)
        weights[step][j] = __bfloat162float(fn[m * wide + b * 1024 + (step * 2 + half) * 128 + col * 8 + j]);
    for (int64_t r = first; r < first + 4 && r < rows; ++r) {
      float acc = 0;
      for (int step = 0; step < 4; ++step) {
        const __nv_bfloat16* in = x + r * wide + b * 1024 + (step * 2 + half) * 128 + col * 8;
        float v = __fmul_rn(__bfloat162float(in[1]), weights[step][1]);
        v = __fmaf_rn(__bfloat162float(in[0]), weights[step][0], v);
        for (int j = 2; j < 8; ++j) v = __fmaf_rn(__bfloat162float(in[j]), weights[step][j], v);
        for (int o = 8; o; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
        acc = __fadd_rn(acc, v);
        acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, v, 16));
      }
      if (lane == 0) part[(r * nb + b) * 32 + m] = acc;
    }
  } else if (blockIdx.z == 0) {
    for (int64_t r = first; r < first + 4 && r < rows; ++r) {
      float v = glm_hc_fast::squares(x + r * wide + b * 1024, lane);
      if (lane == 0) part[(r * nb + b) * 32 + 24] = v;
    }
  }
}

// Prompt-only BF16 tensor products with one FP32 chain per dot; the norm square sum keeps the legacy order.
extern "C" __global__ void __launch_bounds__(128) tf_glm_hc_partial_mma(
    const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb, int64_t rows) {
  using namespace nvcuda;
  __shared__ __align__(32) __nv_bfloat16 a[64 * 64];
  __shared__ __align__(32) __nv_bfloat16 b[32 * 64];
  __shared__ __align__(32) float result[64 * 32];
  const int t = threadIdx.x, warp = t / 32, lane = t % 32;
  const int64_t first = (int64_t)blockIdx.x * 64;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf0, bf1;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
  wmma::fill_fragment(c0, 0.0f);
  wmma::fill_fragment(c1, 0.0f);
  for (int k = 0; k < 1024; k += 64) {
    const int64_t base = (int64_t)blockIdx.y * 1024 + k;
    for (int i = t; i < 64 * 64; i += 128) {
      const int64_t r = first + i / 64;
      a[i] = r < rows ? x[r * wide + base + i % 64] : __float2bfloat16(0);
    }
    for (int i = t; i < 32 * 64; i += 128)
      b[i] = i / 64 < 24 ? fn[(int64_t)(i / 64) * wide + base + i % 64] : __float2bfloat16(0);
    __syncthreads();
    for (int step = 0; step < 64; step += 16) {
      wmma::load_matrix_sync(af, a + warp * 16 * 64 + step, 64);
      wmma::load_matrix_sync(bf0, b + step, 64);
      wmma::load_matrix_sync(bf1, b + 16 * 64 + step, 64);
      wmma::mma_sync(c0, af, bf0, c0);
      wmma::mma_sync(c1, af, bf1, c1);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(result + warp * 16 * 32, c0, 32, wmma::mem_row_major);
  wmma::store_matrix_sync(result + warp * 16 * 32 + 16, c1, 32, wmma::mem_row_major);
  __syncthreads();
  for (int i = t; i < 64 * 32; i += 128) {
    const int64_t r = first + i / 32;
    if (r < rows && i % 32 < 24) part[(r * nb + blockIdx.y) * 32 + i % 32] = result[i];
  }
  for (int q = 0; q < 16; ++q) {
    const int64_t r = first + warp * 16 + q;
    if (r < rows) {
      float v = glm_hc_fast::squares(x + r * wide + blockIdx.y * 1024, lane);
      if (lane == 0) part[(r * nb + blockIdx.y) * 32 + 24] = v;
    }
  }
}
