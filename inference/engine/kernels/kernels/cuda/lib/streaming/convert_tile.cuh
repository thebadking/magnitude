// Expert streaming's per-tile source→resident conversion for the bank-fill
// and convert-wide kernels. Included once per conversion (gate/up/down)
// via bind_gate.cuh, bind_up.cuh, bind_down.cuh with different CONV_*
// macros set before each inclusion.
//
// Required macros set by the includer (bind_*.cuh):
//   CONV_FUNC          — name of the __device__ function to emit
//   CONV_N, CONV_K     — matrix dimensions (rows, columns-total)
//   CONV_SRC_STRIDE_{0,1,2}, CONV_DST_STRIDE_{0,1,2}
//   CONV_SRC_PACKET_SIZE, CONV_SRC_PACKET_GROUP
//   CONV_LAYOUT_PACKET / CONV_LAYOUT_ROWS16 / CONV_LAYOUT_MMA16
//   CONV_CODE_BITS
//   CONV_REPACK_*      — format flags (same as repack_weight.cu REPACK_*)
//   For packet layout: CONV_DST_PACKET_SIZE, CONV_DST_PLANE_{n}_{*}
//   For rows layout:   CONV_DST_ROW_STRIDE_BYTES, CONV_DST_ROW_GROUPS,
//                      CONV_DST_PLANE_*_ROW_OFFSET, _BYTES_PER_ROW
//
// No include guard — designed for multiple inclusion.

// ---- Token-paste helpers -------------------------------------------------
#define CONV_PASTE3_(a, b, c) a##b##c
#define CONV_PASTE3(a, b, c) CONV_PASTE3_(a, b, c)
#define CONV_H(name) CONV_PASTE3(CONV_FUNC, _, name)

// ---- IQ3S grid (guarded: only defined once per TU) ----------------------
#if defined(CONV_REPACK_IQ3S) && !defined(MAGNITUDE_IQ3S_GRID_DEFINED)
#define MAGNITUDE_IQ3S_GRID_DEFINED 1
__constant__ u32 iq3s_grid[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};
#endif

// ---- Grid entry (IQ3XXS / IQ2XS) ----------------------------------------
#if defined(CONV_REPACK_GRID)
__device__ __forceinline__ u32 CONV_H(grid_entry)(const u8 *in, u32 e) {
#if defined(CONV_REPACK_IQ3XXS)
    const u32 block = e / 4, group = e % 4;
    const u32 signs = (u32)in[66 + 4 * block] | ((u32)in[67 + 4 * block] << 8) |
                      ((u32)in[68 + 4 * block] << 16) | ((u32)in[69 + 4 * block] << 24);
    return (u32)in[2 + 8 * block + 2 * group] | ((u32)in[3 + 8 * block + 2 * group] << 8) |
           (((signs >> (7 * group)) & 127u) << 16);
#else
    return (u32)in[2 + 2 * e] | ((u32)in[3 + 2 * e] << 8);
#endif
}
#endif

// ---- Source code reader --------------------------------------------------
__device__ __forceinline__ u32 CONV_H(source_code)(const u8 *in, u32 p) {
#if defined(CONV_REPACK_GRID)
    return (CONV_H(grid_entry)(in, p / 8) >> (CONV_CODE_BITS * (p % 8))) & ((1u << CONV_CODE_BITS) - 1u);
#elif defined(CONV_REPACK_KQUANT) && CONV_CODE_BITS == 4u
    return (in[16 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
#elif defined(CONV_REPACK_KQUANT)
    u32 low  = (in[48 + (p / 64) * 32 + p % 32] >> ((p % 64 / 32) * 4)) & 15u;
    u32 high = (in[16 + p % 32] >> (p / 32)) & 1u;
    return low | (high << 4);
#elif defined(CONV_REPACK_Q6K)
    u32 low  = (in[(p / 128) * 64 + p % 64] >> ((p % 128 / 64) * 4)) & 15u;
    u32 high = (in[128 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    return low | (high << 4);
#elif defined(CONV_REPACK_Q3K)
    u32 low  = (in[32 + (p / 128) * 32 + p % 32] >> ((p % 128 / 32) * 2)) & 3u;
    u32 high = (in[p % 32] >> (p / 32)) & 1u;
    return (low | (high << 2)) + 28u;
#elif defined(CONV_REPACK_IQ3S)
    u32 block = p / 32, group = p % 32 / 8, slot = p % 8;
    u32 point = (u32)in[2 + 8 * block + 2 * group + slot / 4] |
                ((((u32)in[66 + block] >> (2 * group + slot / 4)) & 1u) << 8);
    u32 magnitude = (iq3s_grid[point] >> (8 * (slot % 4))) & 255u;
    return ((in[74 + 4 * block + group] >> slot) & 1u) != 0 ? 32u - magnitude : 32u + magnitude;
#elif defined(CONV_REPACK_Q8)
    return (u32)in[2 + p];
#elif defined(CONV_REPACK_IQ4NL)
    return (in[(p / 32) * 18 + 2 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#elif defined(CONV_REPACK_BLOCK32) || defined(CONV_REPACK_NVFP4)
#if defined(CONV_REPACK_BLOCK32)
    u32 code = (in[CONV_REPACK_QS + p % 16] >> ((p / 16) * 4)) & 15u;
#else
    u32 code = (in[4 + 8 * (p / 16) + p % 8] >> ((p % 16 / 8) * 4)) & 15u;
#endif
#if defined(CONV_REPACK_HIGH_BIT)
    code |= (((u32)in[CONV_REPACK_SCALE_BYTES + p / 8] >> (p % 8)) & 1u) << 4;
#endif
#if defined(CONV_REPACK_E2M1)
    code = code == 8u ? 0u : code;
#endif
    return code;
#else
    return (in[8 + (p / 32) * 16 + p % 16] >> ((p % 32 / 16) * 4)) & 15u;
#endif
}

// ---- K-quant scale helpers -----------------------------------------------
#if defined(CONV_REPACK_KQUANT)
__device__ __forceinline__ u32 CONV_H(kq_local)(const u8 *in, u32 j, u32 field) {
    u32 idx  = j % 4;
    u32 low  = in[4 + field * 4 + idx];
    u32 high = in[12 + idx];
    return j < 4 ? (low & 63u) : (((high >> (4 * field)) & 15u) | ((low >> 6) << 4));
}
__device__ __forceinline__ u32 CONV_H(kq_word)(const u8 *in, u32 w) {
    u32 value = 0;
    for (u32 field = 0; field < 16; ++field) {
        int shift = (int)(field * 6) - (int)(w * 32);
        if (shift <= -6 || shift >= 32) continue;
        u32 local = CONV_H(kq_local)(in, field / 2, field % 2);
        value |= shift >= 0 ? local << shift : local >> (-shift);
    }
    return value;
}
#endif

// ---- IQ4 scale helper ----------------------------------------------------
#if defined(CONV_REPACK_IQ4)
__device__ __forceinline__ float CONV_H(iq4_scale)(const u8 *in, u32 j) {
    float base = seismic_f16_to_f32((unsigned short)(in[0] | (in[1] << 8)));
    u32 low  = (in[4 + j / 2] >> ((j % 2) * 4)) & 15u;
    u32 high = ((u32)(in[2] | (in[3] << 8)) >> (2 * j)) & 3u;
    return __fmul_rn(base, (float)((int)(low | (high << 4)) - 32));
}
#endif

// ---- Scale and super writers ---------------------------------------------
__device__ __forceinline__ void CONV_H(write_scales)(const u8 *in, u8 *out) {
#if defined(CONV_REPACK_KQUANT)
    for (u32 w = 0; w < 3; ++w) reinterpret_cast<u32 *>(out)[w] = CONV_H(kq_word)(in, w);
#elif defined(CONV_REPACK_Q6K)
    for (u32 w = 0; w < 4; ++w)
        reinterpret_cast<u32 *>(out)[w] = (u32)in[192 + 4*w] | ((u32)in[193 + 4*w] << 8) |
                                          ((u32)in[194 + 4*w] << 16) | ((u32)in[195 + 4*w] << 24);
#elif defined(CONV_REPACK_Q3K)
    for (u32 j = 0; j < 16; ++j) {
        u32 low = (in[96 + j % 8] >> (4 * (j / 8))) & 15u;
        u32 high = (in[104 + j % 4] >> (2 * (j / 4))) & 3u;
        out[j] = (u8)((int)(low | (high << 4)) - 32);
    }
#elif defined(CONV_REPACK_IQ3S)
    for (u32 j = 0; j < 16; ++j)
        out[j] = (u8)(1u + 2u * ((in[106 + j / 4] >> (4 * (j / 2 % 2))) & 15u));
#elif defined(CONV_REPACK_IQ3XXS)
    for (u32 j = 0; j < 4; ++j)
        out[j] = (u8)((in[69 + 8*j] >> 4) | (in[73 + 8*j] & 0xf0u));
#elif defined(CONV_REPACK_GRID)
    for (u32 j = 0; j < 8; ++j) out[j] = in[66 + j];
#endif
}

__device__ __forceinline__ void CONV_H(write_supers)(const u8 *in, u8 *out) {
#if defined(CONV_REPACK_KQUANT)
    reinterpret_cast<u32 *>(out)[0] = (u32)in[0] | ((u32)in[1] << 8) | ((u32)in[2] << 16) | ((u32)in[3] << 24);
#elif defined(CONV_REPACK_Q6K)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[208] | (in[209] << 8));
#elif defined(CONV_REPACK_Q3K)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[108] | (in[109] << 8));
#elif defined(CONV_REPACK_IQ3S)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[0] | (in[1] << 8));
#elif defined(CONV_REPACK_Q8)
    reinterpret_cast<unsigned short *>(out)[0] = (unsigned short)(in[0] | (in[1] << 8));
#elif defined(CONV_REPACK_GRID)
    reinterpret_cast<float *>(out)[0] =
        seismic_f16_to_f32((unsigned short)(in[0] | (in[1] << 8))) * CONV_REPACK_GRID_FACTOR;
#elif defined(CONV_REPACK_IQ4NL)
    for (u32 j = 0; j < 8; ++j)
        reinterpret_cast<float *>(out)[j] =
            seismic_f16_to_f32((unsigned short)(in[18*j] | (in[18*j + 1] << 8)));
#elif defined(CONV_REPACK_SCALE_BYTES)
    for (u32 i = 0; i < CONV_REPACK_SCALE_BYTES; ++i) out[i] = in[i];
#else
    for (u32 j = 0; j < 8; ++j)
        reinterpret_cast<float *>(out)[j] = CONV_H(iq4_scale)(in, j);
#endif
}

// ---- Code packing helpers ------------------------------------------------
__device__ __forceinline__ u32 CONV_H(pack_eight)(const u8 *in, u32 first, u32 shift, u32 bits) {
    u32 value = 0;
    for (u32 i = 0; i < 8; ++i)
        value |= ((CONV_H(source_code)(in, first + i) >> shift) & ((1u << bits) - 1u)) << (i * bits);
    return value;
}

__device__ __forceinline__ u32 CONV_H(code_word)(const u8 *in, u32 first, u32 word) {
    u32 value = 0;
    for (u32 code  = (word * 32) / CONV_CODE_BITS;
         code     <= (word * 32 + 31) / CONV_CODE_BITS; ++code) {
        int shift = (int)(code * CONV_CODE_BITS) - (int)(word * 32);
        u32 raw   = CONV_H(source_code)(in, first + code);
        value    |= shift >= 0 ? raw << shift : raw >> (-shift);
    }
    return value;
}

__device__ __forceinline__ u32 CONV_H(pack_four)(const u8 *in, u32 first) {
    u32 value = 0;
    for (u32 i = 0; i < 4; ++i) value |= CONV_H(source_code)(in, first + i) << (8 * i);
    return value;
}

__device__ __forceinline__ void CONV_H(zero_bytes)(u8 *out, u64 count) {
    for (u64 i = 0; i < count; ++i) out[i] = 0;
}

// ---- Source packet address (window-relative: src_b=0, stride0=0) ---------
// The window copy placed packet `pkt` of row `n` at:
//   window + n * CONV_SRC_STRIDE_1 * CONV_SRC_PACKET_SIZE + pkt * CONV_SRC_PACKET_SIZE
__device__ __forceinline__ const u8 *CONV_H(src_pkt)(const u8 *window, u64 n, u64 pkt) {
    return window + (n * CONV_SRC_STRIDE_1 + pkt) * CONV_SRC_PACKET_SIZE;
}

// ---- Destination packet/row address (slot-relative) ----------------------
#if defined(CONV_LAYOUT_PACKET)
__device__ __forceinline__ u8 *CONV_H(dst_pkt)(u8 *dst, u64 slot, u64 n, u64 pkt) {
    return dst + (slot * CONV_DST_STRIDE_0 + n * CONV_DST_STRIDE_1 + pkt * CONV_DST_STRIDE_2) *
                 CONV_DST_PACKET_SIZE;
}
#else
__device__ __forceinline__ u8 *CONV_H(dst_row)(u8 *dst, u64 slot, u64 n) {
    return dst + (slot * CONV_DST_STRIDE_0 + n * CONV_DST_STRIDE_1) * CONV_DST_ROW_STRIDE_BYTES;
}
#if defined(CONV_LAYOUT_MMA16)
__device__ __forceinline__ u8 *CONV_H(tile_plane_byte)(u8 *dst, u64 slot, u64 row0,
                                                        u64 position, u64 row_bytes, u64 offset) {
    u8 *row = CONV_H(dst_row)(dst, slot, row0 + position / row_bytes);
    return row + offset + position % row_bytes;
}
#endif
#endif

// ---- Window-based tile conversion ----------------------------------------
// Converts one 16-row x 32-column tile from the pre-copied shared window.
// window    — shared memory buffer, row-major with CONV_SRC_STRIDE_1-packet strides
// dst       — bank destination buffer
// slot      — bank slot index (dest batch)
// row0      — first row of this tile (row_tile * 16)
// column0   — first column of this tile (tile_x * 32)
// lane      — thread index within the block (0..31 for 32-thread blocks)
// is_last_tile, stored_columns — for rows16 tail zeroing
__device__ void CONV_FUNC(const u8 *window, u8 *dst,
                          u64 slot, u64 row0, u32 column0, u32 lane,
                          bool is_last_tile, u64 stored_columns) {
    const u32 GROUP     = (u32)CONV_SRC_PACKET_GROUP;
    const u32 TILE_ROWS = 16u;
    const u32 TILE_COLS = 32u;
    const u64 packets   = ((u64)CONV_K + GROUP - 1) / GROUP;

#if defined(CONV_LAYOUT_PACKET)
    if (column0 >= packets * GROUP) return;
    const u32 packet     = column0 / GROUP;
    const u32 within     = column0 % GROUP;
    const u32 code_words = CONV_CODE_BITS;
    for (u32 item = lane; item < TILE_ROWS * code_words; item += TILE_COLS) {
        u64 n = row0 + item / code_words;
        if (n >= (u64)CONV_N) continue;
        const u8 *in = CONV_H(src_pkt)(window, n % TILE_ROWS, packet);
        u32 word  = within / 32 * code_words + item % code_words;
        u32 value = CONV_H(code_word)(in, 0, word);
        u8 *out   = CONV_H(dst_pkt)(dst, slot, n, packet);
#if CONV_DST_PACKET_SIZE % 4 == 0
        reinterpret_cast<u32 *>(out + CONV_DST_PLANE_0_OFFSET)[word] = value;
#else
        for (u32 byte = 0; byte < 4; ++byte)
            out[CONV_DST_PLANE_0_OFFSET + 4 * word + byte] = (u8)(value >> (8 * byte));
#endif
    }
    if (within != 0 || lane >= TILE_ROWS || row0 + lane >= (u64)CONV_N) return;
    {
        const u64 n  = row0 + lane;
        const u8 *in = CONV_H(src_pkt)(window, n % TILE_ROWS, packet);
        u8 *out      = CONV_H(dst_pkt)(dst, slot, n, packet);
#if defined(CONV_REPACK_KQUANT) || defined(CONV_REPACK_Q6K) || defined(CONV_REPACK_Q3K) || \
    defined(CONV_REPACK_IQ3S)   || defined(CONV_REPACK_GRID)
        CONV_H(write_scales)(in, out + CONV_DST_PLANE_1_OFFSET);
        CONV_H(write_supers)(in, out + CONV_DST_PLANE_2_OFFSET);
#else
        CONV_H(write_supers)(in, out + CONV_DST_PLANE_1_OFFSET);
#endif
        for (u32 byte = 0; byte < CONV_DST_PACKET_SIZE; ++byte) {
            bool inside = (byte >= CONV_DST_PLANE_0_OFFSET &&
                           byte < CONV_DST_PLANE_0_OFFSET + CONV_DST_PLANE_0_BYTES_PER_GROUP);
            inside = inside || (byte >= CONV_DST_PLANE_1_OFFSET &&
                                byte < CONV_DST_PLANE_1_OFFSET + CONV_DST_PLANE_1_BYTES_PER_GROUP);
#if CONV_DST_PLANE_COUNT > 2
            inside = inside || (byte >= CONV_DST_PLANE_2_OFFSET &&
                                byte < CONV_DST_PLANE_2_OFFSET + CONV_DST_PLANE_2_BYTES_PER_GROUP);
#endif
#if CONV_DST_PLANE_COUNT > 3
            inside = inside || (byte >= CONV_DST_PLANE_3_OFFSET &&
                                byte < CONV_DST_PLANE_3_OFFSET + CONV_DST_PLANE_3_BYTES_PER_GROUP);
#endif
            if (!inside) out[byte] = 0;
        }
    }

#elif defined(CONV_LAYOUT_ROWS16) || defined(CONV_LAYOUT_MMA16)
    if (column0 >= stored_columns) return;
    const bool occupied = column0 < packets * GROUP;
    const u32 within    = column0 % GROUP;
    const u64 rows = (u64)CONV_N - row0 < TILE_ROWS ? (u64)CONV_N - row0 : TILE_ROWS;
#if defined(CONV_LAYOUT_ROWS16)
    if (lane < rows) {
        const u64 n  = row0 + lane;
        const u8 *in = CONV_H(src_pkt)(window, n % TILE_ROWS, column0 / GROUP);
        u8 *row      = CONV_H(dst_row)(dst, slot, n);
#if defined(CONV_DST_PLANE_GRIDS)
        u32 *grids = reinterpret_cast<u32 *>(
            row + CONV_DST_PLANE_GRIDS_ROW_OFFSET + column0 / 32 * 4 * CONV_CODE_BITS);
        for (u32 word = 0; word < CONV_CODE_BITS; ++word)
            grids[word] = CONV_H(code_word)(in, within, word);
#elif defined(CONV_DST_PLANE_CODES)
        uint4 *codes = reinterpret_cast<uint4 *>(row + CONV_DST_PLANE_CODES_ROW_OFFSET + column0);
        codes[0] = make_uint4(CONV_H(pack_four)(in, within), CONV_H(pack_four)(in, within + 4),
                              CONV_H(pack_four)(in, within + 8), CONV_H(pack_four)(in, within + 12));
        codes[1] = make_uint4(CONV_H(pack_four)(in, within + 16), CONV_H(pack_four)(in, within + 20),
                              CONV_H(pack_four)(in, within + 24), CONV_H(pack_four)(in, within + 28));
#else
        *reinterpret_cast<uint4 *>(row + CONV_DST_PLANE_CODES_LO_ROW_OFFSET + column0 / 2) =
            make_uint4(CONV_H(pack_eight)(in, within,      0, 4),
                       CONV_H(pack_eight)(in, within + 8,  0, 4),
                       CONV_H(pack_eight)(in, within + 16, 0, 4),
                       CONV_H(pack_eight)(in, within + 24, 0, 4));
#if defined(CONV_DST_PLANE_CODES_HI) && CONV_DST_PLANE_CODES_HI_CODE_BITS == 1
        *reinterpret_cast<u32 *>(row + CONV_DST_PLANE_CODES_HI_ROW_OFFSET + column0 / 8) =
            CONV_H(pack_eight)(in, within, 4, 1) | (CONV_H(pack_eight)(in, within + 8, 4, 1) << 8) |
            (CONV_H(pack_eight)(in, within + 16, 4, 1) << 16) | (CONV_H(pack_eight)(in, within + 24, 4, 1) << 24);
#elif defined(CONV_DST_PLANE_CODES_HI)
        *reinterpret_cast<uint2 *>(row + CONV_DST_PLANE_CODES_HI_ROW_OFFSET + column0 / 4) =
            make_uint2(CONV_H(pack_eight)(in, within,      4, 2) | (CONV_H(pack_eight)(in, within + 8, 4, 2) << 16),
                       CONV_H(pack_eight)(in, within + 16, 4, 2) | (CONV_H(pack_eight)(in, within + 24, 4, 2) << 16));
#endif
#endif
    }
#endif
    if (within == 0 && lane < rows) {
        const u64 n   = row0 + lane;
        const u64 grp = column0 / GROUP;
        const bool real = n < (u64)CONV_N && occupied;
        u8 *row = CONV_H(dst_row)(dst, slot, n);
#if defined(CONV_DST_PLANE_SCALES)
        u8 *scales = row + CONV_DST_PLANE_SCALES_ROW_OFFSET + grp * CONV_DST_PLANE_SCALES_BYTES_PER_GROUP;
        if (real) CONV_H(write_scales)(CONV_H(src_pkt)(window, n % TILE_ROWS, grp), scales);
        else CONV_H(zero_bytes)(scales, CONV_DST_PLANE_SCALES_BYTES_PER_GROUP);
#endif
        u8 *supers = row + CONV_DST_PLANE_SUPERS_ROW_OFFSET + grp * CONV_DST_PLANE_SUPERS_BYTES_PER_GROUP;
        if (real) CONV_H(write_supers)(CONV_H(src_pkt)(window, n % TILE_ROWS, grp), supers);
        else CONV_H(zero_bytes)(supers, CONV_DST_PLANE_SUPERS_BYTES_PER_GROUP);
    }
    if (is_last_tile && lane < rows) {
        u8 *row = CONV_H(dst_row)(dst, slot, row0 + lane);
#if defined(CONV_DST_PLANE_CODES_HI)
#if defined(CONV_DST_PLANE_SCALES)
        const u64 after_high = CONV_DST_PLANE_SCALES_ROW_OFFSET;
#else
        const u64 after_high = CONV_DST_PLANE_SUPERS_ROW_OFFSET;
#endif
        CONV_H(zero_bytes)(row + CONV_DST_PLANE_CODES_HI_ROW_OFFSET + CONV_DST_PLANE_CODES_HI_BYTES_PER_ROW,
                           after_high - CONV_DST_PLANE_CODES_HI_ROW_OFFSET - CONV_DST_PLANE_CODES_HI_BYTES_PER_ROW);
#endif
#if defined(CONV_DST_PLANE_SCALES)
        CONV_H(zero_bytes)(row + CONV_DST_PLANE_SCALES_ROW_OFFSET + CONV_DST_PLANE_SCALES_BYTES_PER_ROW,
                           CONV_DST_PLANE_SUPERS_ROW_OFFSET - CONV_DST_PLANE_SCALES_ROW_OFFSET -
                               CONV_DST_PLANE_SCALES_BYTES_PER_ROW);
#endif
        CONV_H(zero_bytes)(row + CONV_DST_PLANE_SUPERS_ROW_OFFSET + CONV_DST_PLANE_SUPERS_BYTES_PER_ROW,
                           CONV_DST_ROW_STRIDE_BYTES - CONV_DST_PLANE_SUPERS_ROW_OFFSET -
                               CONV_DST_PLANE_SUPERS_BYTES_PER_ROW);
    }
#endif
}

// ---- Copy a window of staged rows to shared memory -----------------------
// Reads staging memory with volatile loads (system-coherent after await_word).
// Copies rows [row0, row0+16) x packets [packet0, packet0+n_packets) into window.
// window layout: row i at window + i * CONV_SRC_STRIDE_1 * CONV_SRC_PACKET_SIZE
__device__ void CONV_H(copy_window)(volatile const u8 *staging, u8 *window,
                                    u64 record, u64 row0, u64 packet0, u32 n_packets,
                                    u32 lane, u32 threads) {
    const u64 row_bytes = (u64)CONV_SRC_STRIDE_1 * CONV_SRC_PACKET_SIZE;
    for (u32 row = 0; row < 16u; ++row) {
        const u64 n = row0 + row;
        if (n >= (u64)CONV_N) break;
        const u64 offset = (record * CONV_STAGED_STRIDE_0 + n * CONV_STAGED_STRIDE_1 + packet0 * CONV_STAGED_STRIDE_2) *
                           CONV_SRC_PACKET_SIZE;
        volatile const u8 *src = staging + offset;
        u8 *dst = window + row * row_bytes;
        const u64 bytes = (u64)n_packets * CONV_SRC_PACKET_SIZE;
        if (offset % 16 == 0) {
            for (u32 i = lane; i < bytes / 16u; i += threads)
                reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<volatile const uint4 *>(src)[i];
            for (u32 i = (u32)(bytes / 16u * 16u) + lane; i < bytes; i += threads)
                dst[i] = src[i];
        } else if (offset % 4 == 0) {
            for (u32 i = lane; i < bytes / 4u; i += threads)
                reinterpret_cast<u32 *>(dst)[i] = reinterpret_cast<volatile const u32 *>(src)[i];
            for (u32 i = (u32)(bytes / 4u * 4u) + lane; i < bytes; i += threads)
                dst[i] = src[i];
        } else {
            for (u32 i = lane; i < bytes; i += threads) dst[i] = src[i];
        }
    }
}

// ---- Cleanup token-paste helpers ----------------------------------------
#undef CONV_PASTE3_
#undef CONV_PASTE3
#undef CONV_H
