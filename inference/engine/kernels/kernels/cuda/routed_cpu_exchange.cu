#include "lib/streaming/cpu_exchange.cuh"
extern "C" __global__ void routed_cpu_exchange(SEISMIC_KERNEL_PARAMS) {
#include "lib/streaming/cpu_exchange_body.inc"
}
