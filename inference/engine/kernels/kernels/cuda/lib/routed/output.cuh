// Decode output (M <= 8); the CUDA form of `metal/routed_output.metal`.
// Block (x, y) works on the output channels of tile group x for y =
// m * (K + 1) + slot (slot < K is choice (m, slot), slot K the shared
// expert) and publishes each projection A-rounded into scratch. The last
// block of a channel tile to arrive combines its channels: each published
// choice projection weighted by its score in slot order, then residual +
// selected + shared * c, the arithmetic of one block carrying every slot in
// turn. Running the projections side by side keeps enough weight streams in
// flight to reach bandwidth.
// - `routed_output` (one row): block (x, slot) projects slot's down
//   projection; the last of the K + 1 blocks of tile x combines.
// - `routed_output_gathered` (several rows): choice (m, slot) projects only
//   when it is its expert's first choice, the down projection of every
//   choice of that expert (`routed::expert_choices`), so each chosen expert
//   streams once; slot K of row 0 projects the shared expert for every row.
//   The last of tile x's M * (K + 1) blocks combines every row.
#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
#include "routed.cuh"

SEISMIC_PROGRAMMATIC_DEPENDENCY

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;

struct PublishEpi {
    float *published;
    __device__ __forceinline__ void operator()(unsigned, projection::u64 n, float value, float) const {
        published[n] = element::Act::round(value);
    }
};



// Each GEMV row's projection A-rounded at its published offset.
struct GatheredPublishEpi {
    float *published;
    const projection::u64 *offsets;
    __device__ __forceinline__ void operator()(unsigned m, projection::u64 n, float value, float) const {
        published[offsets[m] + n] = element::Act::round(value);
    }
};
