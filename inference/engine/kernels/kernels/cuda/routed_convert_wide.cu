// Expert streaming's scratch fill convert (`routed_convert_wide` in
// routed.seismic): converts the listed E-format staging records into bank
// slots (U-format), using windowed shared-memory copies to avoid per-thread
// system-coherent loads. When list[0] matches this step's sequence, count =
// list[1] records are processed; record i goes from staging list[2+2*i] into
// slot list[3+2*i]. A staged record < SEISMIC_DIM_R comes from `source`,
// one >= SEISMIC_DIM_R from `conveyor` with offset record - DIM_R.
// The CUDA form of metal/routed_convert_wide.metal.

#include "lib/streaming/control.cuh"

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

#include "lib/streaming/bind_convert.cuh"

// Window capacity helpers (same as routed_bank_fill.cu).
static __device__ __forceinline__ u32 conv_window_packets(u32 pkt_size, u32 group) {
    u32 block_pkts = group >= 64u ? 1u : 64u / group;
    u32 fit        = 16384u / (16u * pkt_size);
    return fit >= block_pkts ? fit - fit % block_pkts : block_pkts;
}

extern "C" __global__ void routed_convert_wide(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ u8 window[];
    __shared__ int header[2];
    __shared__ int entry[2];

    volatile const u8 *source    = reinterpret_cast<volatile const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    volatile const u8 *conveyor  = reinterpret_cast<volatile const u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_CONVEYOR));
    u8               *destination = reinterpret_cast<u8 *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATION));
    volatile const int *list      = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST));
    const int        *step        = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));

    const u32 lane    = (u32)threadIdx.x;
    const u32 threads = (u32)blockDim.x;

    if (lane == 0) {
        header[0] = list[0];
        header[1] = list[1];
    }
    __syncthreads();
    if (header[0] != step[0]) return;

    const u64 own         = (u64)SEISMIC_DIM_R;
    const int count       = header[1] < 0 ? 0 : header[1];
    const u32 pkt_size    = SEISMIC_ELEMENT_E_PACKET_SIZE;
    const u32 group       = SEISMIC_ELEMENT_E_LOGICAL_GROUP;
    const u32 win_pkts    = conv_window_packets(pkt_size, group);
    const u64 n           = (u64)SEISMIC_DIM_N;
    const u64 k           = (u64)SEISMIC_DIM_K;
    const u64 packets     = (k + group - 1) / group;
    const u64 row_tiles   = (n + 15) / 16;
    const u64 col_wins    = (packets + win_pkts - 1) / win_pkts;
    const u64 per_record  = row_tiles * col_wins;
    const u64 total_items = (u64)count * per_record;

    // Grid flat index and stride.
    const u64 flat_block  = (u64)blockIdx.x + (u64)gridDim.x * blockIdx.y;
    const u64 flat_blocks = (u64)gridDim.x * gridDim.y;

#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)
    const u64 stored_cols = 0; // unused for packet layout
#else
    const u64 stored_cols = (u64)SEISMIC_DESTINATION_ROW_GROUPS * group;
#endif

    for (u64 item = flat_block; item < total_items; item += flat_blocks) {
        const u64 rec_idx  = item / per_record;
        const u64 win_item = item % per_record;

        if (lane == 0) {
            entry[0] = list[2 + 2 * (int)rec_idx];
            entry[1] = list[3 + 2 * (int)rec_idx];
        }
        __syncthreads();

        const int staged_raw = entry[0];
        const int slot_idx   = entry[1];
        if (staged_raw < 0 || slot_idx < 0) continue;

        const u64 staged  = (u64)staged_raw;
        const u64 record  = staged < own ? staged : staged - own;
        volatile const u8 *src = staged < own ? source : conveyor;

        const u64 row_tile = win_item / col_wins;
        const u64 win_idx  = win_item % col_wins;
        const u64 row0     = row_tile * 16;
        const u64 packet0  = win_idx * win_pkts;
        const u32 n_pkts   = (u32)min((u64)win_pkts,
                                      packets > packet0 ? packets - packet0 : 0ull);
        __syncthreads();
        if (n_pkts > 0)
            conv_convert_copy_window(src, window, record, row0, packet0, n_pkts, lane, threads, seismic_words_value);
        __syncthreads();
        if (n_pkts > 0) {
            const u64 tiles_x = (u64)win_pkts * group / 32;
            // Independent column tiles use every warp of the copied window.
            for (u64 tile = lane / 32u; tile < tiles_x; tile += threads / 32u) {
                const u32 col0    = (u32)(packet0 * group + tile * 32);
                const bool is_last = (win_idx + 1 == col_wins) && (col0 + 32 == stored_cols);
                conv_convert(window, destination, (u64)slot_idx, row0, col0, lane % 32u, is_last, stored_cols, packet0, seismic_words_value);
            }
        }
    }
}
