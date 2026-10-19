// One channel per ordered device queue. A failed channel is never reused:
// later launches must not overwrite input a late host consumer still reads.
#include "lib/streaming/control.cuh"

extern "C" __global__ void routed_cpu_exchange(SEISMIC_KERNEL_PARAMS) {
    seismic_dependency_start();
    const unsigned tid = threadIdx.x;
    const unsigned short *normalized = reinterpret_cast<const unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_NORMALIZED));
    volatile unsigned short *activation = reinterpret_cast<volatile unsigned short *>(SEISMIC_PTR(SEISMIC_BUFFER_ACTIVATION));
    const volatile float *contribution = reinterpret_cast<const volatile float *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTRIBUTION));
    volatile int *control = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));
    const volatile int *heartbeat = reinterpret_cast<const volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_HEARTBEAT));
    const int *step = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    float *value = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    __shared__ int sequence, ok;
    if (tid == 0) {
        streaming::system_fence();
        sequence = step[0];
        ok = control[streaming::FAILURE] == 0 && sequence > 0
            && sequence > control[streaming::REQUEST]
            && sequence > control[streaming::ACKNOWLEDGED];
        if (!ok && control[streaming::FAILURE] == 0) {
            control[streaming::FAILURE] = sequence > 0 ? sequence : -1;
            streaming::system_fence();
        }
    }
    __syncthreads();
    if (ok) {
        for (unsigned long long column = tid; column < SEISMIC_DIM_H; column += blockDim.x)
            activation[column * SEISMIC_ACTIVATION_STRIDE_1] = normalized[column * SEISMIC_NORMALIZED_STRIDE_1];
        // Every producer fences its own writes before thread 0 publishes.
        streaming::system_fence();
        __syncthreads();
        if (tid == 0) {
            control[streaming::REQUEST] = sequence;
            streaming::system_fence();
            int beat = *heartbeat;
            for (unsigned still = 0; still < streaming::PATIENCE; ++still) {
                streaming::system_fence();
                if (control[streaming::ACKNOWLEDGED] == sequence || control[streaming::FAILURE] != 0) break;
                const int now = *heartbeat;
                if (now != beat) { beat = now; still = 0; }
            }
            streaming::system_fence();
            ok = control[streaming::ACKNOWLEDGED] == sequence && control[streaming::FAILURE] == 0;
            if (!ok && control[streaming::FAILURE] == 0) {
                control[streaming::FAILURE] = sequence;
                streaming::system_fence();
            }
        }
        __syncthreads();
    }
    streaming::system_fence();
    for (unsigned long long column = tid; column < SEISMIC_DIM_H; column += blockDim.x)
        value[column * SEISMIC_RESULT_0_STRIDE_1] = ok ? contribution[column * SEISMIC_CONTRIBUTION_STRIDE_1] : 0.0f;
}
