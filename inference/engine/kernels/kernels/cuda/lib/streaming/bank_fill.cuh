// Expert streaming: a streamed layer's absent experts (`routed_bank_fill` in
// routed.seismic), between its route and the resident `routed_gate_up`.
// blockIdx.y = choice, blockIdx.x = tile group (0..31).
// A hit binds its slot and returns. A miss: thread 0 waits for the host's
// placement acknowledgement under the heartbeat watchdog, reads the slot and
// staging records, broadcasts to the block; each block independently copies
// its windowed staging rows to shared memory and converts gate/up/down records
// into the slot. The first tile group stamps and commits.
// The CUDA form of metal/routed_bank_fill.metal.

#include "control.cuh"

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

// Emit conv_gate, conv_up, conv_down and their copy_window helpers.
#include "bind_gate.cuh"
#include "bind_up.cuh"
#include "bind_down.cuh"

// Window capacity: how many source packets fit in 16384 bytes across 16 rows.
// (Matches Metal's Window<B>::packets().)
static __device__ __forceinline__ u32 window_packets(u32 pkt_size, u32 group) {
    u32 block_pkts = group >= 64u ? 1u : 64u / group;
    u32 fit        = 16384u / (16u * pkt_size);
    return fit >= block_pkts ? fit - fit % block_pkts : block_pkts;
}

// Total number of items (row_tile × column_window) for a record with N rows,
// K columns, packet group G, and packet size P.
static __device__ __forceinline__ u64 record_items(u64 n, u64 k, u32 group, u32 pkt_size) {
    const u64 packets    = (k + group - 1) / group;
    const u32 win_pkts   = window_packets(pkt_size, group);
    const u64 row_tiles  = (n + 15) / 16;
    const u64 col_wins   = (packets + win_pkts - 1) / win_pkts;
    return row_tiles * col_wins;
}
