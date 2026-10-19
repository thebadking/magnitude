// Observe the routed GEMV accumulator before its BF16 publication.
#define KERNEL_W0 SEISMIC_EXPERT_WEIGHT
#include "lib/routed/routed.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;

#ifdef SEISMIC_FORMING_ROUTED_PROJECTION
template <int TPW, int KSPLIT>
__global__ void routed_projection(SEISMIC_KERNEL_PARAMS) {
    using Shape = projection::GemvShape<4, TPW, KSPLIT, 1>;
    __shared__ projection::GemvShared<Shape, Pro> shared;
    const projection::u64 group = Shape::tile_group();
    if (group >= projection::gemv_groups<Shape>(SEISMIC_DIM_F))
        return;
    const projection::u64 m = blockIdx.y / SEISMIC_DIM_K, k = blockIdx.y % SEISMIC_DIM_K;
    const int *routes = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const projection::u64 expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
    const Pro pro{SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED) + m * SEISMIC_NORMALIZED_STRIDE_0 * 2,
                  SEISMIC_NORMALIZED_STRIDE_0, projection::AllRows{}};
    const projection::Store<element::F32> epi{
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER) +
            (m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1) * 4, 0, 0};
    projection::gemv_segment<Shape>(shared, pro, 1u, SEISMIC_DIM_H / 64, group, SEISMIC_DIM_F,
                                    KERNEL_W0_MATRIX(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_WEIGHT), expert),
                                    projection::NoWeight{}, epi);
}
#endif
