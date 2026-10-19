#define MAGNITUDE_CPU_EXPERT 1
#include "lib/routed/gate_up.cuh"
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP_CPU
template <int TPW, int KSPLIT>
__global__ void routed_gate_up_cpu(SEISMIC_KERNEL_PARAMS) {
#include "lib/routed/gate_up_body.inc"
}
#endif
