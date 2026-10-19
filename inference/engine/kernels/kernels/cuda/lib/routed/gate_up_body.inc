    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u8 *normalized = SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED);
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 group = Shape::tile_group();
#ifdef MAGNITUDE_CPU_EXPERT
    const projection::u64 cpu_m = blockIdx.y / SEISMIC_DIM_K, cpu_k = blockIdx.y % SEISMIC_DIM_K;
    const int *cpu_routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    if (cpu_routes[cpu_m * SEISMIC_ROUTES_STRIDE_0 + cpu_k * SEISMIC_ROUTES_STRIDE_1] < 0) {
        seismic_dependency_start();
        constexpr projection::u64 CHANNELS = Shape::GROUPS * TPW * 16;
        unsigned short *out = reinterpret_cast<unsigned short *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
        const projection::u64 first = blockIdx.x * CHANNELS;
        for (projection::u64 n = first + threadIdx.x; n < first + CHANNELS && n < SEISMIC_DIM_F; n += blockDim.x)
            out[cpu_m * SEISMIC_RESULT_0_STRIDE_0 + cpu_k * SEISMIC_RESULT_0_STRIDE_1 + n * SEISMIC_RESULT_0_STRIDE_2] = 0;
        return;
    }
#endif
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
        return;
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    const projection::u64 expert = (projection::u64)routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const Pro pro{normalized + m * SEISMIC_NORMALIZED_STRIDE_0 * 2, SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    const Epi epi{SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) + (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 2,
                  0, (int)SEISMIC_PARAM_ACTIVATION};
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE), expert),
                                    KERNEL_W1_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP), expert), epi);
