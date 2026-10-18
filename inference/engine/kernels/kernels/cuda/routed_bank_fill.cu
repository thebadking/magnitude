// Expert streaming: a streamed layer's absent experts (`routed_bank_fill` in
// routed.seismic), between its route and the resident `routed_gate_up`.
// blockIdx.y = choice, blockIdx.x = tile group (0..31).
// A hit binds its slot and returns. A miss: thread 0 waits for the host's
// placement acknowledgement under the heartbeat watchdog, reads the slot and
// staging records, broadcasts to the block; each block independently copies
// its windowed staging rows to shared memory and converts gate/up/down records
// into the slot. The first tile group stamps and commits.
// The CUDA form of metal/routed_bank_fill.metal.

#include "lib/streaming/control.cuh"

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

// Emit conv_gate, conv_up, conv_down and their copy_window helpers.
#include "lib/streaming/bind_gate.cuh"
#include "lib/streaming/bind_up.cuh"
#include "lib/streaming/bind_down.cuh"

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

extern "C" __global__ void routed_bank_fill(SEISMIC_KERNEL_PARAMS) {
    // 16384 bytes shared per block: one window for whichever conversion is in flight.
    extern __shared__ u8 window[];
    __shared__ int words[5]; // [0]=ok, [1]=slot, [2]=gate_record, [3]=up_record, [4]=down_record

    const int *slots    = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOTS));
    volatile int *stamps    = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_STAMPS));
    const int *step         = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    volatile int *control   = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_CONTROL));
    volatile const int *placement = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PLACEMENT));
    volatile const int *heartbeat = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_HEARTBEAT));
    volatile const u8 *staged_gate = reinterpret_cast<volatile const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_STAGED_GATE));
    volatile const u8 *staged_up   = reinterpret_cast<volatile const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_STAGED_UP));
    volatile const u8 *staged_down = reinterpret_cast<volatile const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_STAGED_DOWN));
    u8 *expert_gate  = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_GATE));
    u8 *expert_up    = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_UP));
    u8 *expert_down  = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_DOWN));
    volatile int *slot_map  = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOT_MAP));
    volatile int *commits   = reinterpret_cast<volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_COMMITS));
    int *bound       = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));

    const u32 choice  = (u32)blockIdx.y;
    const u32 group_x = (u32)blockIdx.x;
    const u32 lane    = (u32)threadIdx.x;
    const u32 threads = (u32)blockDim.x;
    const u64 m = choice / (u64)SEISMIC_DIM_K;
    const u64 k = choice % (u64)SEISMIC_DIM_K;

    // Resolved slot for this choice from the pre-route slot map.
    const int resolved = slots[m * SEISMIC_SLOTS_STRIDE_0 + k * SEISMIC_SLOTS_STRIDE_1];
    if (resolved >= 0) {
        if (group_x == 0 && lane == 0)
            bound[m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1] = resolved;
        return;
    }

    const int sequence = step[0];

    // Thread 0: wait for host placement under heartbeat watchdog.
    if (lane == 0) {
        const u64 col = (u64)choice * SEISMIC_PLACEMENT_STRIDE_1;
        const bool acked = streaming::await_word(
            const_cast<volatile const int *>(placement + 4ull * SEISMIC_PLACEMENT_STRIDE_0 + col),
            control + streaming::FAILURE, heartbeat, sequence);
        words[0] = acked ? 1 : 0;
        if (acked) {
            for (int row = 0; row < 4; ++row)
                words[1 + row] = placement[(u64)row * SEISMIC_PLACEMENT_STRIDE_0 + col];
        }
    }
    __syncthreads();

    // After a failure, use slot 0 to keep addresses in bounds; host discards results.
    const int slot     = words[0] != 0 ? words[1] : 0;
    if (group_x == 0 && lane == 0)
        bound[m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1] = slot;
    if (words[0] == 0 || words[2] < 0)
        return;

    // The union assigns one slot and record triple per expert; only the first
    // requested choice of that expert converts it (others defer to its work).
    const int expert = control[streaming::EXPERTS + choice];
    for (u32 prior = 0; prior < choice; ++prior) {
        const u64 at = (u64)prior * SEISMIC_PLACEMENT_STRIDE_1;
        const int prior_slot = slots[(prior / SEISMIC_DIM_K) * SEISMIC_SLOTS_STRIDE_0 +
                                     (prior % SEISMIC_DIM_K) * SEISMIC_SLOTS_STRIDE_1];
        if (prior_slot < 0 && control[streaming::EXPERTS + prior] == expert &&
            placement[at] == slot && placement[SEISMIC_PLACEMENT_STRIDE_0 + at] >= 0)
            return;
    }

    const u64 gate_record = (u64)words[2];
    const u64 up_record   = (u64)words[3];
    const u64 down_record = (u64)words[4];

    // ---- Gate conversion ------------------------------------------------
    // Window params: (pkt_size, group) are compile-time constants.
    static_assert(SEISMIC_ELEMENT_SG_LOGICAL_GROUP > 0, "gate group must be positive");
    const u32 gate_pkt_size  = SEISMIC_ELEMENT_SG_PACKET_SIZE;
    const u32 gate_group     = SEISMIC_ELEMENT_SG_LOGICAL_GROUP;
    const u32 gate_win_pkts  = window_packets(gate_pkt_size, gate_group);
    const u64 gate_n         = SEISMIC_DIM_F;
    const u64 gate_k         = SEISMIC_DIM_H;
    const u64 gate_packets   = (gate_k + gate_group - 1) / gate_group;
    const u64 gate_row_tiles = (gate_n + 15) / 16;
    const u64 gate_col_wins  = (gate_packets + gate_win_pkts - 1) / gate_win_pkts;
    const u64 gate_items     = gate_row_tiles * gate_col_wins;
#if defined(SEISMIC_ELEMENT_EGW_LAYOUT_PACKET)
    const u64 gate_stored_cols = 0; // unused for packet layout
#else
    const u64 gate_stored_cols = SEISMIC_EXPERT_GATE_ROW_GROUPS * gate_group;
#endif

    for (u64 item = group_x; item < gate_items; item += 32u) {
        const u32 row_tile = (u32)(item / gate_col_wins);
        const u32 win_idx  = (u32)(item % gate_col_wins);
        const u64 row0     = (u64)row_tile * 16;
        const u64 packet0  = (u64)win_idx * gate_win_pkts;
        const u32 n_pkts   = (u32)min((u64)gate_win_pkts, gate_packets > packet0 ? gate_packets - packet0 : 0ull);
        __syncthreads();
        if (n_pkts > 0)
            conv_gate_copy_window(staged_gate, window, gate_record, row0, packet0, n_pkts, lane, threads, seismic_words_value);
        __syncthreads();
        if (n_pkts > 0) {
            // Convert tiles in this window: each tile covers 32 columns = 1 column group.
            const u64 tiles_x = (gate_col_wins > 0) ? (u64)gate_win_pkts * gate_group / 32 : 0;
            for (u64 tile = lane / 32u; tile < tiles_x; tile += threads / 32u) {
                const u32 col0    = (u32)(packet0 * gate_group + tile * 32);
                const bool is_last = (win_idx + 1 == gate_col_wins) && (col0 + 32 == gate_stored_cols);
                conv_gate(window, expert_gate, (u64)slot, row0, col0, lane % 32u, is_last, gate_stored_cols, packet0, seismic_words_value);
            }
        }
    }

    // ---- Up conversion --------------------------------------------------
    const u32 up_pkt_size  = SEISMIC_ELEMENT_SU_PACKET_SIZE;
    const u32 up_group     = SEISMIC_ELEMENT_SU_LOGICAL_GROUP;
    const u32 up_win_pkts  = window_packets(up_pkt_size, up_group);
    const u64 up_n         = SEISMIC_DIM_F;
    const u64 up_k         = SEISMIC_DIM_H;
    const u64 up_packets   = (up_k + up_group - 1) / up_group;
    const u64 up_row_tiles = (up_n + 15) / 16;
    const u64 up_col_wins  = (up_packets + up_win_pkts - 1) / up_win_pkts;
    const u64 up_items     = up_row_tiles * up_col_wins;
#if defined(SEISMIC_ELEMENT_EUW_LAYOUT_PACKET)
    const u64 up_stored_cols = 0;
#else
    const u64 up_stored_cols = SEISMIC_EXPERT_UP_ROW_GROUPS * up_group;
#endif

    for (u64 item = group_x; item < up_items; item += 32u) {
        const u32 row_tile = (u32)(item / up_col_wins);
        const u32 win_idx  = (u32)(item % up_col_wins);
        const u64 row0     = (u64)row_tile * 16;
        const u64 packet0  = (u64)win_idx * up_win_pkts;
        const u32 n_pkts   = (u32)min((u64)up_win_pkts, up_packets > packet0 ? up_packets - packet0 : 0ull);
        __syncthreads();
        if (n_pkts > 0)
            conv_up_copy_window(staged_up, window, up_record, row0, packet0, n_pkts, lane, threads, seismic_words_value);
        __syncthreads();
        if (n_pkts > 0) {
            const u64 tiles_x = (up_col_wins > 0) ? (u64)up_win_pkts * up_group / 32 : 0;
            for (u64 tile = lane / 32u; tile < tiles_x; tile += threads / 32u) {
                const u32 col0    = (u32)(packet0 * up_group + tile * 32);
                const bool is_last = (win_idx + 1 == up_col_wins) && (col0 + 32 == up_stored_cols);
                conv_up(window, expert_up, (u64)slot, row0, col0, lane % 32u, is_last, up_stored_cols, packet0, seismic_words_value);
            }
        }
    }

    // ---- Down conversion ------------------------------------------------
    const u32 dn_pkt_size  = SEISMIC_ELEMENT_SD_PACKET_SIZE;
    const u32 dn_group     = SEISMIC_ELEMENT_SD_LOGICAL_GROUP;
    const u32 dn_win_pkts  = window_packets(dn_pkt_size, dn_group);
    const u64 dn_n         = SEISMIC_DIM_H;   // down is transposed: [H, F]
    const u64 dn_k         = SEISMIC_DIM_F;
    const u64 dn_packets   = (dn_k + dn_group - 1) / dn_group;
    const u64 dn_row_tiles = (dn_n + 15) / 16;
    const u64 dn_col_wins  = (dn_packets + dn_win_pkts - 1) / dn_win_pkts;
    const u64 dn_items     = dn_row_tiles * dn_col_wins;
#if defined(SEISMIC_ELEMENT_EDW_LAYOUT_PACKET)
    const u64 dn_stored_cols = 0;
#else
    const u64 dn_stored_cols = SEISMIC_EXPERT_DOWN_ROW_GROUPS * dn_group;
#endif

    for (u64 item = group_x; item < dn_items; item += 32u) {
        const u32 row_tile = (u32)(item / dn_col_wins);
        const u32 win_idx  = (u32)(item % dn_col_wins);
        const u64 row0     = (u64)row_tile * 16;
        const u64 packet0  = (u64)win_idx * dn_win_pkts;
        const u32 n_pkts   = (u32)min((u64)dn_win_pkts, dn_packets > packet0 ? dn_packets - packet0 : 0ull);
        __syncthreads();
        if (n_pkts > 0)
            conv_down_copy_window(staged_down, window, down_record, row0, packet0, n_pkts, lane, threads, seismic_words_value);
        __syncthreads();
        if (n_pkts > 0) {
            const u64 tiles_x = (dn_col_wins > 0) ? (u64)dn_win_pkts * dn_group / 32 : 0;
            for (u64 tile = lane / 32u; tile < tiles_x; tile += threads / 32u) {
                const u32 col0    = (u32)(packet0 * dn_group + tile * 32);
                const bool is_last = (win_idx + 1 == dn_col_wins) && (col0 + 32 == dn_stored_cols);
                conv_down(window, expert_down, (u64)slot, row0, col0, lane % 32u, is_last, dn_stored_cols, packet0, seismic_words_value);
            }
        }
    }

    if (group_x != 0 || lane != 0) return;

    stamps[(u64)slot * SEISMIC_STAMPS_STRIDE_0] = sequence;

    // Commit only after a successful, acknowledged request.
    if (!streaming::await_acknowledgement(control, heartbeat, sequence) ||
        control[streaming::FAILURE] == sequence ||
        expert < 0 || (u64)expert >= SEISMIC_DIM_X)
        return;

    slot_map[(u64)expert * SEISMIC_SLOT_MAP_STRIDE_0] = slot;
    streaming::system_fence();
    commits[(u64)slot * SEISMIC_COMMITS_STRIDE_0] = sequence;
    streaming::system_fence();
}
