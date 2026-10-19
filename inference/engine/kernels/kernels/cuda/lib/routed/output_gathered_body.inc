    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, routed::GatheredRows> shared;
    __shared__ projection::u64 rows_at[8], published_at[8];
    __shared__ unsigned count, last;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K, M = SEISMIC_DIM_M;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    float *published = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED));
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));

    // The routes come from the launch before the expansion, which has
    // completed (the expansion lets this launch start only after its own
    // wait), so every block reads them and issues its first weight loads
    // before waiting.
    const unsigned choice = (unsigned)(m * K + slot);
    const bool projects = slot < K
        ? routed::first_choice(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice)
        : m == 0;
    if (projects && threadIdx.x == 0) {
        if (slot < K) {
            projection::u32 chosen[8];
            count = routed::expert_choices(routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)M,
                                           (unsigned)K, choice, chosen);
            for (unsigned r = 0; r < count; ++r) {
                const projection::u64 row = chosen[r] / K, choice_slot = chosen[r] % K;
                rows_at[r] = row * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + choice_slot * SEISMIC_EXPERT_PRODUCT_STRIDE_1;
                published_at[r] = (row * (K + 1) + choice_slot) * H;
            }
        } else {
            count = (unsigned)M;
            for (unsigned r = 0; r < count; ++r) {
                rows_at[r] = r * SEISMIC_SHARED_PRODUCT_STRIDE_0;
                published_at[r] = (r * (K + 1) + K) * H;
            }
        }
    }
    __syncthreads();
    if (projects) {
        const GatheredPublishEpi epi{published, published_at};
        if (slot < K) {
            const projection::u64 expert = (projection::u64)routed::route(
                routes, SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1, (unsigned)K, choice);
            const routed::GatheredRows rows{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT), rows_at};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,
                                                  SEISMIC_DIM_F / 64, group, H,
                                                  KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                                  projection::NoWeight{}, epi);
        } else {
            const routed::GatheredRows rows{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT), rows_at};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(rows), count,
                                                  SEISMIC_DIM_S / 64, group, H,
                                                  KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                                  projection::NoWeight{}, epi);
        }
    } else {
        seismic_dependency_start();
    }

    // Arrive (sync scratch: zero when the launch starts, restored by the last).
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned *counter = arrivals + blockIdx.x;
        const bool is_last = atomicAdd(counter, 1u) == gridDim.y - 1;
        if (is_last) {
            __threadfence();
            atomicExch(counter, 0u);
        }
        last = is_last;
    }
    __syncthreads();
    if (!last)
        return;

    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const float *coefficients = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT));
    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const float *inject = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT));
    const projection::u64 first = blockIdx.x * CHANNELS;
    for (projection::u64 item = threadIdx.x; item < M * CHANNELS; item += blockDim.x) {
        const projection::u64 row = item / CHANNELS, n = first + item % CHANNELS;
        if (n >= H)
            continue;
        const float *row_published = published + row * (K + 1) * H;
        float selected = 0.0f;
        for (projection::u64 k = 0; k < K; ++k)
            selected = __fmaf_rn(scores[row * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1],
                                 __ldcg(row_published + k * H + n), selected);
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(
            out + row * SEISMIC_RESULT_0_STRIDE_0, residual + row * SEISMIC_RESIDUAL_STRIDE_0,
            inject + row * SEISMIC_INJECT_STRIDE_0, H, n, selected,
            __ldcg(row_published + K * H + n) * coefficients[row * SEISMIC_COEFFICIENT_STRIDE_0]);
    }
