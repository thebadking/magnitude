// Expert streaming, the step sequence (`routed_step` in routed.seismic): one
// thread sets it to the step's own sequence, which the host wrote before
// submitting the step. Every later entry of the step reads it in order on
// the same stream, so no system fence is needed — stream ordering is enough.

extern "C" __global__ void routed_step(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x == 0) {
        int *step              = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
        const int *sequence    = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEQUENCE));
        step[0] = sequence[0];
    }
}
