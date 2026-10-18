// Expert streaming, steps 1 and 2 (`routed_route_streamed` in routed.seismic):
// a streamed layer's routing GEMV, selection, slot resolve, control publication,
// shared expert GEMV, and prefetch hint GEMV + hint publication in one launch.
// Block dispatch (blockIdx.x):
//   [0, route_blocks)                — routing GEMV per expert column; the last
//                                      block to arrive selects, resolves, and
//                                      (if any choice is absent) publishes the
//                                      streaming request under system fences.
//   [route_blocks, +shared_blocks)   — shared-expert gate/up GEMV.
//   [route_blocks+shared_blocks, +2*predict_blocks) — near (first half) and far
//                                      (second half) prefetch router GEMV; the
//                                      last block per distance publishes hints.
// CUDA form of metal/routed_route_streamed.metal.

#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include "lib/streaming/control.cuh"
#define KERNEL_W0 SEISMIC_ROUTER
#define KERNEL_W1 SEISMIC_SHARED_GATE
#define KERNEL_W2 SEISMIC_SHARED_UP
#include <seismic/packets.cuh>

namespace {
using element::Act;
typedef ELEMENT_OF(SEISMIC_NORM) Norm;
#if !defined(SEISMIC_ROUTER_KIND_PACKED)
typedef ELEMENT_OF(SEISMIC_ROUTER) Router;
#endif
} // namespace

SEISMIC_PROGRAMMATIC_DEPENDENCY

constexpr unsigned STREAM_THREADS = 256u;
constexpr unsigned STREAM_WARPS   = STREAM_THREADS / 32u;
constexpr unsigned ROUTE_BLOCKS   = (unsigned)((SEISMIC_DIM_E + STREAM_WARPS - 1) / STREAM_WARPS);
// Near/far each use ROUTE_BLOCKS blocks.
constexpr unsigned PREDICT_BLOCKS = ROUTE_BLOCKS;

// ---- Helpers shared with routed_route.cu --------------------------------
static __device__ __forceinline__ bool routed_stream_precedes(float a, int ae, float b, int be) {
    return a > b || (a == b && ae > be);
}

static __device__ __forceinline__ float routed_stream_warp_max(float v) {
#pragma unroll
    for (unsigned mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, mask));
    return v;
}

static __device__ __forceinline__ float routed_stream_warp_sum(float v) {
#pragma unroll
    for (unsigned mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    return v;
}

// The declared input form: an RMS-normalized row, or the sum of already
// normalized mix partials, with one activation rounding at publication.
static __device__ __forceinline__ float stream_normed(
        const float *residual, const unsigned char *norm,
        unsigned long long row, unsigned long long source, float inverse,
        const seismic_words_t &seismic_words_value) {
    if constexpr (SEISMIC_DIM_NM == 0) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned long long part = 0; part < SEISMIC_DIM_PT; ++part)
            sum = seismic_add_rn(sum, residual[row * SEISMIC_RESIDUAL_STRIDE_0 +
                (part * SEISMIC_DIM_H + source) * SEISMIC_RESIDUAL_STRIDE_1]);
        return Act::round(sum);
    } else {
        const float v = residual[row * SEISMIC_RESIDUAL_STRIDE_0 + source * SEISMIC_RESIDUAL_STRIDE_1];
        const float g = element::at<Norm>(norm, source * SEISMIC_NORM_STRIDE_0);
        return Act::round(seismic_mul_rn(seismic_mul_rn(v, inverse), g));
    }
}

// Decode one packed row in the registry's 16-value fragment order.
template <class W>
static __device__ __forceinline__ float stream_packed_dot(
        const float *residual, const unsigned char *norm, const W &weight,
        unsigned long long feature, unsigned long long row, float inverse,
        const seismic_words_t &seismic_words_value) {
    float dot = 0.0f;
    const unsigned lane = threadIdx.x % 32u;
    for (unsigned item = lane; item < ((SEISMIC_DIM_H + 63) / 64) * 4; item += 32) {
        const unsigned kb = item / 4, t = item % 4;
        const unsigned offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
        float values[16];
        packets::row_values16(weight, feature, kb, t, values);
#pragma unroll
        for (unsigned step = 0; step < 4; ++step)
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned source = 64 * kb + 16 * step + offsets[j];
                if (source < SEISMIC_DIM_H)
                    dot = seismic_fma_rn(stream_normed(residual, norm, row, source, inverse, seismic_words_value),
                        values[4 * step + j], dot);
            }
    }
    return dot;
}

// Selection of `row` from the logits scratch, K-argmax in probability space.
// Writes routes, scores, coefficient; returns true for lane 0.
static __device__ __forceinline__ void stream_select_row(
        const float *logits, unsigned long long row, unsigned lane,
        int *routes, float *scores, float *coefficient,
        bool normalize, const seismic_words_t &seismic_words_value) {
    constexpr unsigned PER_LANE = (unsigned)SEISMIC_DIM_E / 32u;
    const float neg_inf = -__int_as_float(0x7f800000);
    float prob[PER_LANE];
    float maximum = neg_inf;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        prob[i] = __ldcg(logits + row * (SEISMIC_DIM_E + 1u) + lane + 32u * i);
        maximum = fmaxf(maximum, prob[i]);
    }
    maximum = routed_stream_warp_max(maximum);
    float total = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) {
        prob[i] = expf(prob[i] - maximum);
        total = seismic_add_rn(total, prob[i]);
    }
    total = routed_stream_warp_sum(total);
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i) prob[i] /= total;

    float chosen[SEISMIC_DIM_K];
    for (unsigned rank = 0; rank < (unsigned)SEISMIC_DIM_K; ++rank) {
        float best = neg_inf;
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i) {
            const int e = (int)(lane + 32u * i);
            if (routed_stream_precedes(prob[i], e, best, best_expert)) {
                best = prob[i];
                best_expert = e;
            }
        }
        float winner_prob;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_prob);
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i)
            if (winner >= 0 && (unsigned)winner == lane + 32u * i)
                prob[i] = neg_inf;
        chosen[rank] = winner_prob;
        if (lane == 0) {
            const unsigned long long slot = SEISMIC_DIM_K - 1u - rank;
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = winner;
            scores[row * SEISMIC_SCORES_STRIDE_0 + slot * SEISMIC_SCORES_STRIDE_1] = winner_prob;
        }
    }
    if (normalize) {
        float denom = 0.0f;
        for (unsigned s = 0; s < (unsigned)SEISMIC_DIM_K; ++s)
            denom = seismic_add_rn(denom, chosen[SEISMIC_DIM_K - 1u - s]);
        if (lane == 0)
            for (unsigned s = 0; s < (unsigned)SEISMIC_DIM_K; ++s)
                scores[row * SEISMIC_SCORES_STRIDE_0 + s * SEISMIC_SCORES_STRIDE_1] =
                    chosen[SEISMIC_DIM_K - 1u - s] / denom;
    }
    if (lane == 0)
        coefficient[row * SEISMIC_RESULT_1_STRIDE_0] =
            1.0f / (1.0f + expf(-__ldcg(logits + row * (SEISMIC_DIM_E + 1u) + SEISMIC_DIM_E)));
}

// Predict the K largest logits for `row` out of `predict_logits` (a M*(E+1)
// slice), writing winners into `chosen[row * K + rank]`.
static __device__ __forceinline__ void stream_predict_row(
        const float *predict_logits, unsigned long long row,
        unsigned lane, int *chosen, const seismic_words_t &seismic_words_value) {
    constexpr unsigned PER_LANE = (unsigned)SEISMIC_DIM_E / 32u;
    const float neg_inf = -__int_as_float(0x7f800000);
    float value[PER_LANE];
#pragma unroll
    for (unsigned i = 0; i < PER_LANE; ++i)
        value[i] = predict_logits[row * (SEISMIC_DIM_E + 1u) + lane + 32u * i];

    for (unsigned rank = 0; rank < (unsigned)SEISMIC_DIM_K; ++rank) {
        float best = neg_inf;
        int best_expert = -1;
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i) {
            const int e = (int)(lane + 32u * i);
            if (routed_stream_precedes(value[i], e, best, best_expert)) {
                best = value[i];
                best_expert = e;
            }
        }
        float winner_val;
        const int winner = reduce::argmax<reduce::HigherIndex>(best, best_expert, winner_val);
#pragma unroll
        for (unsigned i = 0; i < PER_LANE; ++i)
            if (winner >= 0 && (unsigned)winner == lane + 32u * i)
                value[i] = neg_inf;
        if (lane == 0 && row < SEISMIC_DIM_M)
            chosen[row * SEISMIC_DIM_K + rank] = winner;
    }
}

// Publish hint for one distance: chosen[rows*K], absent from map.
static __device__ __forceinline__ void stream_publish_hint(
        const int *chosen, const volatile int *map, unsigned long long map_stride,
        volatile int *ring, unsigned long long ring_stride,
        int sequence, unsigned rows, unsigned tid) {
    if (tid != 0) return;
    const unsigned total = rows * (unsigned)SEISMIC_DIM_K;
    int count = 0;
    for (unsigned c = 0; c < total; ++c) {
        const int expert = chosen[c];
        if (expert < 0 || map[(unsigned long long)expert * map_stride] >= 0) continue;
        bool duplicate = false;
        for (unsigned earlier = 0; earlier < c; ++earlier)
            duplicate |= chosen[earlier] == expert;
        if (!duplicate)
            ring[(2 + (unsigned long long)count++) * ring_stride] = expert;
    }
    ring[1 * ring_stride] = count;
    streaming::system_fence();
    ring[0] = sequence;
    streaming::system_fence();
}

extern "C" __global__ void routed_route_streamed(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    // Scratch region layout: [0,M*(E+1)) current layer; [M*(E+1),2M*(E+1)) near; [2M*(E+1),3M*(E+1)) far.
    float *logits      = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_LOGITS));
    volatile unsigned int *arrivals = reinterpret_cast<volatile unsigned int *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));

    const float *residual     = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const unsigned char *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const unsigned char *router = SEISMIC_PTR(SEISMIC_BUFFER_ROUTER);
    int *routes = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    float *scores = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const volatile int *slot_map  = reinterpret_cast<const volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOT_MAP));
    volatile int *stamps          = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_STAMPS));
    const int *step               = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    volatile int *control         = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));
    const unsigned char *near_norm   = SEISMIC_PTR(SEISMIC_BUFFER_NEAR_NORM);
    const unsigned char *near_router = SEISMIC_PTR(SEISMIC_BUFFER_NEAR_ROUTER);
    const volatile int *near_map     = reinterpret_cast<const volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_NEAR_MAP));
    const unsigned char *far_norm    = SEISMIC_PTR(SEISMIC_BUFFER_FAR_NORM);
    const unsigned char *far_router  = SEISMIC_PTR(SEISMIC_BUFFER_FAR_ROUTER);
    const volatile int *far_map      = reinterpret_cast<const volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_FAR_MAP));
    volatile int *near_ring  = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_NEAR_RING));
    volatile int *far_ring   = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_FAR_RING));
    unsigned char *normalized  = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *coefficient         = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    unsigned char *shared_product = SEISMIC_PTR(SEISMIC_RESULT_2_BUFFER);
    int *slots                 = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_3_BUFFER));
    const unsigned char *shared_gate = SEISMIC_PTR(SEISMIC_BUFFER_SHARED_GATE);
    const unsigned char *shared_up   = SEISMIC_PTR(SEISMIC_BUFFER_SHARED_UP);
    const float *shared_router = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_ROUTER));

    const unsigned block  = (unsigned)blockIdx.x;
    const unsigned tid    = (unsigned)threadIdx.x;
    const unsigned lane   = tid % 32u;
    const unsigned warp   = tid / 32u;
    const unsigned rows   = (unsigned)SEISMIC_DIM_M;
    const float eps       = __uint_as_float((unsigned)SEISMIC_PARAM_EPS);
    const int sequence    = step[0];
    const bool normalize  = SEISMIC_PARAM_NORMALIZE != 0;
    const int ahead       = (int)SEISMIC_PARAM_AHEAD;
    const unsigned long long plane = (unsigned long long)rows * (SEISMIC_DIM_E + 1u);

    // ---- Compute RMS inverses (all blocks, all rows) ---------------------
    __shared__ float inverses[8];   // at most 8 decode rows
    for (unsigned row = warp; row < rows; row += STREAM_WARPS) {
        if constexpr (SEISMIC_DIM_NM != 0) {
            float sq = 0.0f;
            for (unsigned s = lane; s < SEISMIC_DIM_H; s += 32u) {
                const float v = residual[(unsigned long long)row * SEISMIC_RESIDUAL_STRIDE_0 +
                                         (unsigned long long)s * SEISMIC_RESIDUAL_STRIDE_1];
                sq = seismic_fma_rn(v, v, sq);
            }
            const float total = routed_stream_warp_sum(sq);
            if (lane == 0) inverses[row] = rsqrtf(seismic_add_rn(total / (float)SEISMIC_DIM_H, eps));
        } else {
            if (lane == 0) inverses[row] = 1.0f;
        }
    }
    __syncthreads();

    // ---- [0, route_blocks): routing GEMV ---------------------------------
    if (block < ROUTE_BLOCKS) {
        // Each warp in the block handles one expert column.
        const unsigned long long expert_base = (unsigned long long)block * STREAM_WARPS;
        const unsigned long long expert      = expert_base + warp;

        // Block 0 publishes the normalized rows.
        if (block == 0 && warp == 0) {
            for (unsigned row = 0; row < rows; ++row) {
                const float inv = inverses[row];
                for (unsigned s = lane; s < SEISMIC_DIM_H; s += 32u) {
                    const float x = stream_normed(residual, norm, row, s, inv, seismic_words_value);
                    element::put<Act>(normalized,
                        (unsigned long long)row * SEISMIC_RESULT_0_STRIDE_0 + (unsigned long long)s * SEISMIC_RESULT_0_STRIDE_1,
                        x);
                }
            }
        }

        // Router dot products for experts [expert_base, expert_base + STREAM_WARPS).
        if (expert < (unsigned long long)SEISMIC_DIM_E) {
            for (unsigned long long row = 0; row < rows; ++row) {
                float dot = 0.0f;
                const float inv = inverses[row];
                #if defined(SEISMIC_ROUTER_KIND_PACKED)
                dot = stream_packed_dot(residual, norm, KERNEL_W0_AT(router), expert, row, inv, seismic_words_value);
                #else
                for (unsigned col = lane; col < SEISMIC_DIM_H; col += 32u) {
                    const float w = element::at<Router>(router, expert * SEISMIC_ROUTER_STRIDE_0 + col);
                    dot = seismic_fma_rn(stream_normed(residual, norm, row, col, inv, seismic_words_value), w, dot);
                }
                #endif
                dot = routed_stream_warp_sum(dot);
                if (lane == 0)
                    logits[row * (SEISMIC_DIM_E + 1u) + expert] = dot;
            }
        }
        // Shared-expert gate logit (column E): dot of normalized row with shared_router.
        if (warp == STREAM_WARPS - 1 || expert_base + warp == (unsigned long long)SEISMIC_DIM_E - 1) {
            if (block == ROUTE_BLOCKS - 1 && warp == STREAM_WARPS - 1) {
                for (unsigned long long row = 0; row < rows; ++row) {
                    float dot = 0.0f;
                    const float inv = inverses[row];
                    for (unsigned s = lane; s < SEISMIC_DIM_H; s += 32u) {
                        const float x = stream_normed(residual, norm, row, s, inv, seismic_words_value);
                        dot = seismic_fma_rn(x, shared_router[s], dot);
                    }
                    dot = routed_stream_warp_sum(dot);
                    if (lane == 0)
                        logits[row * (SEISMIC_DIM_E + 1u) + SEISMIC_DIM_E] = dot;
                }
            }
        }
        __threadfence();
        __syncthreads();

        // Last block to arrive: select, then resolve.
        __shared__ bool last_route;
        if (tid == 0) {
            const unsigned prev = atomicAdd(const_cast<unsigned int *>(&arrivals[0]), 1u);
            last_route = prev + 1u == ROUTE_BLOCKS;
            if (last_route) atomicExch(const_cast<unsigned int *>(&arrivals[0]), 0u);
        }
        __syncthreads();
        if (!last_route) return;

        // Selection.
        for (unsigned long long row = warp; row < rows; row += STREAM_WARPS)
            stream_select_row(logits, row, lane, routes, scores, coefficient, normalize, seismic_words_value);
        __threadfence();
        __syncthreads();

        // Resolve: look up slot_map, write slots, stamp.
        __shared__ int missing, invalid;
        if (tid == 0) { missing = 0; invalid = 0; }
        __syncthreads();
        const unsigned choices = rows * (unsigned)SEISMIC_DIM_K;
        for (unsigned c = tid; c < choices; c += STREAM_THREADS) {
            const unsigned long long m      = c / SEISMIC_DIM_K;
            const unsigned long long choice = c % SEISMIC_DIM_K;
            const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1];
            if (expert < 0 || (unsigned)expert >= (unsigned)SEISMIC_DIM_E) {
                // Nonfinite routing cannot index the directory. Bound later
                // consumers safely; the failure prevents state publication.
                slots[m * SEISMIC_RESULT_3_STRIDE_0 + choice * SEISMIC_RESULT_3_STRIDE_1] = 0;
                atomicOr(&invalid, 1);
                continue;
            }
            const int slot   = slot_map[(unsigned long long)expert * SEISMIC_SLOT_MAP_STRIDE_0];
            slots[m * SEISMIC_RESULT_3_STRIDE_0 + choice * SEISMIC_RESULT_3_STRIDE_1] = slot;
            if (slot >= 0)
                stamps[(unsigned long long)slot * SEISMIC_STAMPS_STRIDE_0] = sequence;
            else
                atomicOr(&missing, 1);
        }
        __threadfence();
        __syncthreads();

        if (invalid != 0) {
            if (tid == 0) {
                control[streaming::FAILURE] = sequence;
                streaming::system_fence();
            }
            return;
        }
        if (missing == 0) return;

        // Publish the request under system-scope fences.
        if (tid == 0) {
            streaming::system_fence();
            for (unsigned c = 0; c < choices; ++c) {
                const unsigned long long m      = c / SEISMIC_DIM_K;
                const unsigned long long choice = c % SEISMIC_DIM_K;
                const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + choice * SEISMIC_ROUTES_STRIDE_1];
                const int slot   = slots[m * SEISMIC_RESULT_3_STRIDE_0 + choice * SEISMIC_RESULT_3_STRIDE_1];
                control[streaming::EXPERTS + c] = slot >= 0 ? -1 - expert : expert;
            }
            control[streaming::COUNT] = (int)choices;
            streaming::system_fence();
            control[streaming::REQUEST] = sequence;
            streaming::system_fence();
        }
        return;
    }

    // ---- [route_blocks, +shared_blocks): shared-expert gate/up GEMV -----
    // (Shared expert uses the already-published normalized rows.)
    // shared_blocks = ceil_div(S, STREAM_WARPS): each warp covers one shared feature.
    const unsigned shared_blocks = ((unsigned)SEISMIC_DIM_S + STREAM_WARPS - 1u) / STREAM_WARPS;
    if (block < ROUTE_BLOCKS + shared_blocks) {
        const unsigned long long feature_base = (unsigned long long)(block - ROUTE_BLOCKS) * STREAM_WARPS + warp;
        if (feature_base < (unsigned long long)SEISMIC_DIM_S) {
            for (unsigned long long row = 0; row < rows; ++row) {
                float gate = 0.0f, up = 0.0f;
                const float inv = inverses[row];
#if defined(SEISMIC_SHARED_GATE_KIND_PACKED)
                gate = stream_packed_dot(residual, norm, KERNEL_W1_AT(shared_gate), feature_base, row, inv, seismic_words_value);
#else
                for (unsigned col = lane; col < SEISMIC_DIM_H; col += 32u)
                    gate = seismic_fma_rn(stream_normed(residual, norm, row, col, inv, seismic_words_value),
                        element::at<ELEMENT_OF(SEISMIC_SHARED_GATE)>(shared_gate, feature_base * SEISMIC_SHARED_GATE_STRIDE_0 + col), gate);
#endif
#if defined(SEISMIC_SHARED_UP_KIND_PACKED)
                up = stream_packed_dot(residual, norm, KERNEL_W2_AT(shared_up), feature_base, row, inv, seismic_words_value);
#else
                for (unsigned col = lane; col < SEISMIC_DIM_H; col += 32u)
                    up = seismic_fma_rn(stream_normed(residual, norm, row, col, inv, seismic_words_value),
                        element::at<ELEMENT_OF(SEISMIC_SHARED_UP)>(shared_up, feature_base * SEISMIC_SHARED_UP_STRIDE_0 + col), up);
#endif
                gate = routed_stream_warp_sum(gate);
                up   = routed_stream_warp_sum(up);
                if (lane == 0)
                    element::put<Act>(shared_product,
                        row * SEISMIC_RESULT_2_STRIDE_0 + feature_base * SEISMIC_RESULT_2_STRIDE_1,
                        Act::round((gate / (1.0f + expf(-gate))) * up));
            }
        }
        return;
    }

    // ---- [route_blocks+shared_blocks, +2*predict_blocks): prefetch GEMV --
    if (ahead <= 0) return;
    const unsigned predict_base = ROUTE_BLOCKS + shared_blocks;
    if (block >= predict_base + 2u * PREDICT_BLOCKS) return;
    const unsigned ahead_idx = (block - predict_base) / PREDICT_BLOCKS;
    const unsigned pred_block = (block - predict_base) % PREDICT_BLOCKS;
    if (ahead_idx >= (unsigned)ahead) return;

    const bool far = ahead_idx == 1u;
    const unsigned char *pred_norm   = far ? far_norm   : near_norm;
    const unsigned char *pred_router = far ? far_router : near_router;
    // Predict logits offset into the scratch: slot (1 + ahead_idx) * plane.
    float *pred_logits = logits + (1u + ahead_idx) * plane;

    // Predict GEMV.
    const unsigned long long pred_expert_base = (unsigned long long)pred_block * STREAM_WARPS;
    const unsigned long long pred_expert = pred_expert_base + warp;
    if (pred_expert < (unsigned long long)SEISMIC_DIM_E) {
        for (unsigned long long row = 0; row < rows; ++row) {
            float dot = 0.0f;
            const float inv = inverses[row];
            #if defined(SEISMIC_ROUTER_KIND_PACKED)
            dot = stream_packed_dot(residual, pred_norm, KERNEL_W0_AT(pred_router), pred_expert, row, inv, seismic_words_value);
            #else
            for (unsigned col = lane; col < SEISMIC_DIM_H; col += 32u) {
                const float w = element::at<Router>(pred_router, pred_expert * SEISMIC_NEAR_ROUTER_STRIDE_0 + col);
                dot = seismic_fma_rn(stream_normed(residual, pred_norm, row, col, inv, seismic_words_value), w, dot);
            }
            #endif
            dot = routed_stream_warp_sum(dot);
            if (lane == 0)
                pred_logits[row * (SEISMIC_DIM_E + 1u) + pred_expert] = dot;
        }
    }
    __threadfence();
    __syncthreads();

    // Arrive; last block for this distance publishes hints.
    __shared__ bool last_prediction;
    if (tid == 0) {
        const unsigned prev = atomicAdd(const_cast<unsigned int *>(&arrivals[1 + ahead_idx]), 1u);
        last_prediction = prev + 1u == PREDICT_BLOCKS;
        if (last_prediction) atomicExch(const_cast<unsigned int *>(&arrivals[1 + ahead_idx]), 0u);
    }
    __syncthreads();
    if (!last_prediction) return;

    // Select K predictions per row.
    __shared__ int chosen[8 * SEISMIC_DIM_K];
    for (unsigned long long row = warp; row < rows; row += STREAM_WARPS)
        stream_predict_row(pred_logits, row, lane, chosen, seismic_words_value);
    __syncthreads();

    // Publish hints to the ring.
    const volatile int *pred_map = far ? far_map : near_map;
    const unsigned long long map_stride = far ? SEISMIC_FAR_MAP_STRIDE_0 : SEISMIC_NEAR_MAP_STRIDE_0;
    volatile int *ring             = far ? far_ring  : near_ring;
    const unsigned long long ring_stride = far ? SEISMIC_FAR_RING_STRIDE_0 : SEISMIC_NEAR_RING_STRIDE_0;
    stream_publish_hint(chosen, pred_map, map_stride, ring, ring_stride, sequence, rows, tid);
}
