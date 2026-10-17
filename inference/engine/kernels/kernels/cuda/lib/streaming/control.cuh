// Expert streaming's device side of the host handshake: system-scope fences
// and the await spin loop, the CUDA form of metal/lib/streaming/control.h.
//
// Words the host reads or writes while a step runs are accessed through
// volatile pointers under __threadfence_system(). In CUDA, __threadfence_system()
// makes writes visible to the host CPU and ensures volatile loads see the
// latest host-written value (expert-streaming spec §4.3).

#pragma once

namespace streaming {

// Words of a layer's request region (routed.seismic).
static constexpr unsigned REQUEST      = 0u;
static constexpr unsigned ACKNOWLEDGED = 1u;
static constexpr unsigned FAILURE      = 2u;
static constexpr unsigned COUNT        = 3u;
static constexpr unsigned EXPERTS      = 4u;

// Words of the load's scratch request region.
static constexpr unsigned SCRATCH_LAYER = 1u;

// How long to spin before declaring a dead host. A live host is never
// abandoned however long its DMA reads take; a dead one never holds
// the device. Matches the Metal PATIENCE constant.
static constexpr unsigned PATIENCE = 1u << 22;

__device__ __forceinline__ void system_fence() {
    __threadfence_system();
}

// Waits until `word` holds `sequence` or `failure` does; on a stalled
// heartbeat writes the failure itself. Returns true when `word` arrived
// and the step has not failed. Only thread 0 of the block calls this.
__device__ __forceinline__ bool await_word(volatile const int *word,
                                           volatile int *failure,
                                           volatile const int *heartbeat,
                                           int sequence) {
    int beat = *heartbeat;
    for (unsigned still = 0; still < PATIENCE; ++still) {
        system_fence();
        if (*word == sequence || *failure == sequence) break;
        const int now = *heartbeat;
        if (now != beat) {
            beat = now;
            still = 0;
        }
    }
    system_fence();
    if (*word != sequence && *failure != sequence) {
        *failure = sequence;
        system_fence();
    }
    return *word == sequence && *failure != sequence;
}

__device__ __forceinline__ bool await_acknowledgement(volatile int *control,
                                                       volatile const int *heartbeat,
                                                       int sequence) {
    return await_word(
        const_cast<volatile const int *>(control + ACKNOWLEDGED),
        control + FAILURE,
        heartbeat,
        sequence);
}

} // namespace streaming
