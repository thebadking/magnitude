// Decode expansion of gated experts (M <= 8); the CUDA form of
// `metal/routed_gate_up.metal`. Block row y = choice (m, k): the K1 paired
// GEMV over expert routes[m, k]'s gate/up rows for one activation row, with
// the activation-generic GLU epilogue.
#define KERNEL_W0 SEISMIC_EXPERT_GATE
#define KERNEL_W1 SEISMIC_EXPERT_UP
#include "routed.cuh"

using Pro = projection::Plain<ELEMENT_OF(SEISMIC_ELEMENT_A), projection::AllRows>;
using Epi = projection::Glu<ELEMENT_OF(SEISMIC_ELEMENT_A)>;
