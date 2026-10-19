#include "lib/routed/gate_up.cuh"
#ifdef SEISMIC_FORMING_ROUTED_GATE_UP
template <int TPW, int KSPLIT>
__global__ void routed_gate_up(SEISMIC_KERNEL_PARAMS) {
#include "lib/routed/gate_up_body.inc"
}
#endif
