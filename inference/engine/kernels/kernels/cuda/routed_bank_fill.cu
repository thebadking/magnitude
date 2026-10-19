#include "lib/streaming/bank_fill.cuh"
extern "C" __global__ void routed_bank_fill(SEISMIC_KERNEL_PARAMS) {
#include "lib/streaming/bank_fill_body.inc"
}
