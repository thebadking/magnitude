#include "lib/streaming/bank_fill.cuh"
extern "C" __global__ void routed_bank_fill_cpu(SEISMIC_KERNEL_PARAMS) {
#include "lib/streaming/bank_fill_body.inc"
}
