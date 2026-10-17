// Expert streaming, one round of a scratch fill (`routed_scratch_wait` in
// routed.seismic): thread 0 publishes the step's sequence at the round's
// arrival word, then waits for it at the round's acknowledgement word or
// the failure word, failing the step itself if the host's heartbeat stalls.
// The CUDA form of metal/routed_scratch_wait.metal.

#include "lib/streaming/control.cuh"

extern "C" __global__ void routed_scratch_wait(SEISMIC_KERNEL_PARAMS) {
    if (threadIdx.x != 0) return;

    const int *step              = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    volatile int *control        = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));
    volatile const int *heartbeat = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HEARTBEAT));

    const int sequence = step[0];
    const unsigned acknowledgement = (unsigned)SEISMIC_PARAM_ACKNOWLEDGEMENT;
    streaming::system_fence();
    control[(unsigned)SEISMIC_PARAM_ARRIVAL] = sequence;
    streaming::system_fence();
    streaming::await_word(
        const_cast<volatile const int *>(control + acknowledgement),
        control + streaming::FAILURE,
        heartbeat,
        sequence);
}
