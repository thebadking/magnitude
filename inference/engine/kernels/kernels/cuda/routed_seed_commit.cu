// Expert streaming, a scratch fill's seeding (`routed_seed_commit` in
// routed.seismic): after `routed_copy` placed the converted experts into
// their bank slots, the listed slots are stamped and, unless the step failed,
// their map entries published under a system-scope fence before their commits.
// The CUDA form of metal/routed_seed_commit.metal.

#include "lib/streaming/control.cuh"

extern "C" __global__ void routed_seed_commit(SEISMIC_KERNEL_PARAMS) {
    volatile const int *list    = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST));
    const int *step             = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    volatile const int *control = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));
    volatile int *slot_map      = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOT_MAP));
    volatile int *stamps        = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_STAMPS));
    volatile int *commits       = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMMITS));

    const int sequence = step[0];
    streaming::system_fence();

    if (list[0] != sequence || control[streaming::FAILURE] == sequence)
        return;

    const int count = min(list[1], (int)SEISMIC_DIM_E);
    const unsigned threads = blockDim.x;
    const unsigned tid     = threadIdx.x;

    // Stamp slots and publish map entries.
    for (int entry = (int)tid; entry < count; entry += (int)threads) {
        const int expert = list[2 + 2 * entry];
        const int slot   = list[3 + 2 * entry];
        if (expert < 0 || (unsigned long long)expert >= SEISMIC_DIM_X ||
            slot   < 0 || (unsigned long long)slot   >= SEISMIC_DIM_BANK)
            continue;
        stamps[(unsigned long long)slot * SEISMIC_STAMPS_STRIDE_0] = sequence;
        slot_map[(unsigned long long)expert * SEISMIC_SLOT_MAP_STRIDE_0] = slot;
    }
    streaming::system_fence();
    __threadfence();

    // Commit after map entries are published.
    for (int entry = (int)tid; entry < count; entry += (int)threads) {
        const int expert = list[2 + 2 * entry];
        const int slot   = list[3 + 2 * entry];
        if (expert < 0 || (unsigned long long)expert >= SEISMIC_DIM_X ||
            slot   < 0 || (unsigned long long)slot   >= SEISMIC_DIM_BANK)
            continue;
        commits[(unsigned long long)slot * SEISMIC_COMMITS_STRIDE_0] = sequence;
    }
    streaming::system_fence();
}
