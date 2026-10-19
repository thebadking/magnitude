    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    __shared__ unsigned last;
    const projection::u64 H = SEISMIC_DIM_H, K = SEISMIC_DIM_K;
    const projection::u64 m = blockIdx.y / (K + 1), slot = blockIdx.y % (K + 1);
    const projection::u64 group = Shape::tile_group();
    float *published = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED)) + m * (K + 1) * H;
    unsigned *arrivals = reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS));

    if (group < projection::gemv_groups<Shape>(H)) {
        const PublishEpi epi{published + slot * H};
        if (slot < K) {
            const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
            const projection::u64 expert =
                (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1];
#ifdef MAGNITUDE_CPU_EXPERT
            if (routes[m * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] < 0) {
                seismic_dependency_start();
                const projection::u64 first = blockIdx.x * CHANNELS;
                for (projection::u64 n = first + threadIdx.x; n < first + CHANNELS && n < H; n += blockDim.x)
                    published[slot * H + n] = element::Act::round(MAGNITUDE_CPU_CONTRIBUTION(m,n));
            } else
#endif
            {
            const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_PRODUCT)
                              + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + slot * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * 2,
                          0, projection::AllRows{}};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_F / 64, group, H,
                                            KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN), expert),
                                            projection::NoWeight{}, epi);
            }
        } else {
            const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_SHARED_PRODUCT) + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * 2, 0,
                          projection::AllRows{}};
            projection::gemv_segment_ready<Shape>(shared, projection::after_dependency(pro), 1u, SEISMIC_DIM_S / 64, group, H,
                                            KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_SHARED_DOWN)),
                                            projection::NoWeight{}, epi);
        }
    } else {
        // Wait before arriving: the arrival and the combine follow the
        // launch before.
        seismic_dependency_start();
    }

    // Arrive (sync scratch: zero when the launch starts, restored by the last).
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned *counter = arrivals + m * gridDim.x + blockIdx.x;
        const bool is_last = atomicAdd(counter, 1u) == K;
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
    const float *residual =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL)) + m * SEISMIC_RESIDUAL_STRIDE_0;
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + m * SEISMIC_RESULT_0_STRIDE_0;
    const float *inject = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INJECT)) + m * SEISMIC_INJECT_STRIDE_0;
    const float coefficient =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENT))[m * SEISMIC_COEFFICIENT_STRIDE_0];
    const projection::u64 first = blockIdx.x * CHANNELS;
    for (projection::u64 n = first + threadIdx.x; n < first + CHANNELS && n < H; n += blockDim.x) {
        float selected = 0.0f;
        for (projection::u64 k = 0; k < K; ++k)
            selected = __fmaf_rn(scores[m * SEISMIC_SCORES_STRIDE_0 + k * SEISMIC_SCORES_STRIDE_1],
                                 __ldcg(published + k * H + n), selected);
        projection::routed_join<SEISMIC_DIM_SN, (SEISMIC_DIM_IN != 0)>(
            out, residual, inject, H, n, selected, __ldcg(published + K * H + n) * coefficient);
    }
