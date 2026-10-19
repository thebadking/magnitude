#include "lib/routed/output.cuh"
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT
template <int TPW, int KSPLIT>
__global__ void routed_output(SEISMIC_KERNEL_PARAMS) {
#include "lib/routed/output_body.inc"
}
#endif
#ifdef SEISMIC_FORMING_ROUTED_OUTPUT_GATHERED
template <int TPW, int KSPLIT>
__global__ void routed_output_gathered(SEISMIC_KERNEL_PARAMS) {
#include "lib/routed/output_gathered_body.inc"
}
#endif
