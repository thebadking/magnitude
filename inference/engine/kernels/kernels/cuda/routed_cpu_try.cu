#define MAGNITUDE_CPU_EXPERT 1
static __device__ __forceinline__ bool cpu_exchange_eligible(const int *slots, const seismic_words_t &seismic_words_value) {
    unsigned missing = 0;
    for (unsigned k = 0; k < SEISMIC_DIM_K; ++k)
        missing += slots[k * SEISMIC_SLOTS_STRIDE_1] < 0;
    return missing == 1;
}
#define MAGNITUDE_CPU_ELIGIBLE cpu_exchange_eligible(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOTS)), seismic_words_value)
#include "lib/streaming/cpu_exchange.cuh"
extern "C" __global__ void routed_cpu_try(SEISMIC_KERNEL_PARAMS) {
#include "lib/streaming/cpu_exchange_body.inc"
}
