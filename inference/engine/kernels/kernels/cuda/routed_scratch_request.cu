// Expert streaming, a scratch-served step's request (`routed_scratch_request`
// in routed.seismic): each expert's flag, the number of choices that route to
// it (0 when none), then the layer and the step's sequence in the load's
// scratch request region, under system-scope fences. It does not wait.
// The CUDA form of metal/routed_scratch_request.metal.

#include "lib/streaming/control.cuh"

extern "C" __global__ void routed_scratch_request(SEISMIC_KERNEL_PARAMS) {
    const int *routes  = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROUTES));
    const int *step    = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    volatile int *flags   = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_FLAGS));
    volatile int *control = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));

    // E <= 1024 (routed.seismic): one shared counter per expert.
    __shared__ int counts[1024];

    const unsigned experts  = (unsigned)SEISMIC_DIM_E;
    const unsigned long long choices = (unsigned long long)SEISMIC_DIM_M * (unsigned long long)SEISMIC_DIM_K;
    const unsigned tid      = threadIdx.x;

    for (unsigned e = tid; e < experts; e += 1024u)
        counts[e] = 0;
    __syncthreads();

    for (unsigned long long c = tid; c < choices; c += 1024u) {
        const unsigned long long m = c / (unsigned long long)SEISMIC_DIM_K;
        const unsigned long long k = c % (unsigned long long)SEISMIC_DIM_K;
        const int expert = routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1];
        atomicAdd(&counts[(unsigned)expert], 1);
    }
    __syncthreads();

    for (unsigned e = tid; e < experts; e += 1024u)
        flags[(unsigned long long)e * SEISMIC_FLAGS_STRIDE_0] = counts[e];

    __threadfence_system();
    __syncthreads();

    if (tid != 0) return;
    __threadfence_system();
    control[streaming::SCRATCH_LAYER] = (int)SEISMIC_PARAM_LAYER;
    __threadfence_system();
    control[streaming::REQUEST] = step[0];
    __threadfence_system();
}
