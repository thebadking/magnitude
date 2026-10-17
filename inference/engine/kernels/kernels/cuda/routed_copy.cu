// Expert streaming, a scratch fill's copy (`routed_copy` in routed.seismic):
// the listed records of the source into the destination, byte for byte. One
// thread reads the list header and entry indices into shared memory; threads
// stride over each matrix in 16-byte vectors, then its tail bytes.
// The CUDA form of metal/routed_copy.metal.

// A matrix's byte stride: a packet layout's matrix stride counts packets;
// a row layout's counts stored rows.
#if defined(SEISMIC_ELEMENT_U_LAYOUT_PACKET)
#define COPY_MATRIX_BYTES ((unsigned long long)(SEISMIC_SOURCE_STRIDE_0) * (unsigned long long)(SEISMIC_ELEMENT_U_PACKET_SIZE))
#else
#define COPY_MATRIX_BYTES ((unsigned long long)(SEISMIC_SOURCE_STRIDE_0) * (unsigned long long)(SEISMIC_SOURCE_ROW_STRIDE_BYTES))
#endif

extern "C" __global__ void routed_copy(SEISMIC_KERNEL_PARAMS) {
    const unsigned char *source    = reinterpret_cast<const unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    volatile const int *list       = reinterpret_cast<volatile const int *>(SEISMIC_PTR(SEISMIC_BUFFER_LIST));
    const int *step                = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STEP));
    unsigned char *destination     = reinterpret_cast<unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATION));

    __shared__ int header[2];
    __shared__ int entry_records[2];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        header[0] = list[0];
        header[1] = list[1];
    }
    __syncthreads();

    if (header[0] != step[0]) return;

    const unsigned long long bytes  = COPY_MATRIX_BYTES;
    // Grid: (X-groups, Y-groups); tid = flat thread index within block.
    const unsigned long long lane   = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long lanes  = (unsigned long long)gridDim.x  * blockDim.x;
    const unsigned long long vectors = bytes / 16ull;
    const int count = header[1];

    for (int entry = (int)blockIdx.y; entry < count; entry += (int)gridDim.y) {
        if (threadIdx.x == 0) {
            entry_records[0] = list[2 + 2 * entry];
            entry_records[1] = list[3 + 2 * entry];
        }
        __syncthreads();
        const unsigned char *from = source      + (unsigned long long)entry_records[0] * bytes;
        unsigned char       *to   = destination + (unsigned long long)entry_records[1] * bytes;
        __syncthreads();
        for (unsigned long long v = lane; v < vectors; v += lanes)
            reinterpret_cast<uint4 *>(to)[v] = reinterpret_cast<const uint4 *>(from)[v];
        for (unsigned long long b = vectors * 16ull + lane; b < bytes; b += lanes)
            to[b] = from[b];
    }
}
