#define MAGNITUDE_CPU_EXPERT 1
#define MAGNITUDE_CPU_CONTRIBUTION(m,n) reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_CPU_CONTRIBUTION))[(m) * SEISMIC_CPU_CONTRIBUTION_STRIDE_0 + (n) * SEISMIC_CPU_CONTRIBUTION_STRIDE_1]
#include "lib/routed/output.cuh"
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_CPU
template <int TPW, int KSPLIT>
__global__ void routed_output_cpu(SEISMIC_KERNEL_PARAMS) {
#include "lib/routed/output_body.inc"
}
#endif
