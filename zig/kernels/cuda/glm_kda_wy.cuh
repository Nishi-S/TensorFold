// Opt-in native WY32 prompt step inspired by Mia 0039; finite FP32 order differs from serial recurrence.
struct KdaWyShared {
    float q[32][128], k[32][128], g[32][128];
    float l[32][32], m[32][32], delta[32][128], beta[32], end[128];
    int nonfinite;
};
static_assert(sizeof(KdaWyShared) == 74372, "WY shared ABI");

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_wy(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in,
        const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
        const float* __restrict__ g_save, const float* __restrict__ b_save, int rows,
        __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out, int offset) {
    extern __shared__ __align__(16) unsigned char raw[];
    KdaWyShared& w = *reinterpret_cast<KdaWyShared*>(raw);
    const int h = blockIdx.x, tid = threadIdx.x, v = tid / 2, side = tid % 2;
    if (tid == 0) w.nonfinite = 0;
    __syncthreads();
    bool bad = false;
    for (int i = tid; i < 128 * 128; i += 256) bad |= !isfinite(state_in[(size_t)h * 128 * 128 + i]);
    for (int i = tid; i < rows * 128; i += 256) {
        const size_t at = ((size_t)(i / 128) * H + h) * 128 + i % 128;
        bad |= !isfinite(q_in[at]) || !isfinite(k_save[at]) || !isfinite(g_save[at]) || !isfinite(__bfloat162float(v_save[at]));
    }
    for (int r = tid; r < rows; r += 256) bad |= !isfinite(b_save[r * H + h]);
    if (bad) atomicExch(&w.nonfinite, 1);
    __syncthreads();
    if (w.nonfinite) {
        const int lane = tid % 32, warp = tid / 32;
        for (int quarter = 0; quarter < 4; ++quarter) {
            const int vbase = quarter * 32 + warp * 4;
            float ss[4][4];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    ss[j][i] = state_in[((size_t)h * 128 + vbase + j) * 128 + lane * 4 + i];
            for (int r = 0; r < rows; ++r) {
                float kk[4], qq[4], gg[4], vv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const size_t at = ((size_t)r * H + h) * 128 + lane * 4 + i;
                    kk[i] = k_save[at]; qq[i] = q_in[at]; gg[i] = g_save[at];
                    vv[i] = __bfloat162float(v_save[((size_t)r * H + h) * 128 + vbase + i]);
                }
                update4(ss, kk, gg, vv, b_save[r * H + h], lane);
                float part[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float out = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) out += ss[j][i] * qq[i];
                    part[j] = out;
                }
                const float out = warp_sum4(part, lane);
                if ((lane & 7) == 0) y_out[((size_t)r * H + h) * 128 + vbase + (lane >> 3)] = __float2bfloat16_rn(out);
            }
            if (state_out != nullptr)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        state_out[((size_t)h * 128 + vbase + j) * 128 + lane * 4 + i] = ss[j][i];
        }
        return;
    }
    float state[64];
    #pragma unroll
    for (int c = 0; c < 64; ++c)
        state[c] = state_in[((size_t)h * 128 + v) * 128 + side * 64 + c];
    for (int first = 0; first < rows;) {
        const int room = first == 0 ? 32 - offset : 32;
        const int n = rows - first < room ? rows - first : room;
        for (int x = tid; x < n * 128; x += 256) {
            const int r = x / 128, c = x % 128;
            const size_t at = ((size_t)(first + r) * H + h) * 128 + c;
            w.q[r][c] = q_in[at]; w.k[r][c] = k_save[at]; w.g[r][c] = g_save[at];
        }
        if (tid < n) w.beta[tid] = b_save[(first + tid) * H + h];
        __syncthreads();
        const int t = tid / 8, lane8 = tid % 8;
        if (t < n) {
            float decay[16];
            #pragma unroll
            for (int c = 0; c < 16; ++c) decay[c] = 1.0f;
            const unsigned mask = 0xffu << ((tid % 32) / 8 * 8);
            for (int row = t; row < n; ++row) {
                float l = 0.0f, m = 0.0f;
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    const int col = lane8 * 16 + c;
                    if (row != t) decay[c] *= w.g[row][col];
                    const float ke = w.k[t][col] * decay[c];
                    l = __fmaf_rn(w.k[row][col], ke, l);
                    m = __fmaf_rn(w.q[row][col], ke, m);
                }
                for (int step = 4; step; step >>= 1) {
                    l += __shfl_xor_sync(mask, l, step, 8);
                    m += __shfl_xor_sync(mask, m, step, 8);
                }
                if (lane8 == 0) {
                    w.l[row][t] = row == t ? 0.0f : w.beta[row] * l;
                    w.m[row][t] = m;
                }
            }
        }
        __syncthreads();
        if (tid < 128) {
            float prefix[32], e = 1.0f;
            #pragma unroll
            for (int r = 0; r < 32; ++r) if (r < n) { e *= w.g[r][tid]; prefix[r] = e; }
            w.end[tid] = e;
            float suffix = 1.0f;
            #pragma unroll
            for (int r = 31; r >= 0; --r) if (r < n) {
                const float decay = w.g[r][tid];
                w.g[r][tid] = w.k[r][tid] * suffix;
                suffix *= decay;
                w.k[r][tid] *= prefix[r]; w.q[r][tid] *= prefix[r];
            }
        }
        __syncthreads();
        for (int r = 0; r < n; ++r) {
            float y = 0.0f;
            #pragma unroll
    for (int c = 0; c < 64; ++c) y = __fmaf_rn(w.k[r][side * 64 + c], state[c], y);
            y += __shfl_xor_sync(0xffffffffu, y, 1);
            if (side == 0) {
                const float value = __bfloat162float(v_save[((size_t)(first + r) * H + h) * 128 + v]);
                y = w.beta[r] * (value - y);
                for (int t = 0; t < r; ++t) y = __fmaf_rn(-w.l[r][t], w.delta[t][v], y);
                w.delta[r][v] = y;
            }
        }
        __syncthreads();
        for (int r = 0; r < n; ++r) {
            float out = 0.0f;
            #pragma unroll
    for (int c = 0; c < 64; ++c) out = __fmaf_rn(w.q[r][side * 64 + c], state[c], out);
            out += __shfl_xor_sync(0xffffffffu, out, 1);
            if (side == 0) {
                for (int t = 0; t <= r; ++t) out = __fmaf_rn(w.m[r][t], w.delta[t][v], out);
                y_out[((size_t)(first + r) * H + h) * 128 + v] = __float2bfloat16_rn(out);
            }
        }
        #pragma unroll
    for (int c = 0; c < 64; ++c) {
            float s = state[c] * w.end[side * 64 + c];
            for (int r = 0; r < n; ++r) s = __fmaf_rn(w.delta[r][v], w.g[r][side * 64 + c], s);
            state[c] = s;
        }
        __syncthreads();
        first += n;
    }
    if (state_out != nullptr)
        #pragma unroll
    for (int c = 0; c < 64; ++c)
            state_out[((size_t)h * 128 + v) * 128 + side * 64 + c] = state[c];
}
