// One CPU expert, with the ordinary routed CPU library's BF16 publications.
// Eight output rows are one rows8 weight tile. The four stages share one
// native submission and never quantize the activation to INT8.
use lib::core::activation;
use lib::projection::projection;
use lib::routed::routed;
use seismic::cpu::Bf16;

fn routed_cpu_expert_stage<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    _group: [u64; 3],
    _shared: &mut [u8],
) {
    let h = cx.dim_h() as usize;
    // SAFETY: this single work item owns the input staging row.
    let staged = unsafe { cx.scratch_input().slice_mut::<f32>(0, h) };
    projection::stage::<Bf16>(cx.arg_normalized().row([0, 0]), staged);
}

fn routed_cpu_expert_up<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (h, f) = (cx.dim_h() as usize, cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], 8, f);
    // SAFETY: input staging is complete; disjoint work items own product tiles.
    let staged = unsafe { cx.scratch_input().slice::<f32>(0, h) };
    let product = unsafe {
        cx.scratch_product()
            .slice_mut::<u16>(2 * rows.start, rows.len())
    };
    routed::glu_into::<Bf16>(
        &cx.arg_expert_gate(),
        &cx.arg_expert_up(),
        rows.start,
        staged,
        None,
        0,
        product,
    );
}

fn routed_cpu_expert_product<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    _group: [u64; 3],
    _shared: &mut [u8],
) {
    let f = cx.dim_f() as usize;
    // SAFETY: every product tile is complete before this single staging item.
    let product = unsafe { cx.scratch_product().slice::<u16>(0, f) };
    let staged = unsafe { cx.scratch_down().slice_mut::<f32>(0, f) };
    projection::stage::<Bf16>(product, staged);
}

fn routed_cpu_expert_down<L: Isa, E: Elements>(
    _l: L,
    cx: &Context<'_, E>,
    group: [u64; 3],
    _shared: &mut [u8],
) {
    let (h, f) = (cx.dim_h() as usize, cx.dim_f() as usize);
    let rows = projection::item_rows(group[0], 8, h);
    // SAFETY: staged product is complete; output tiles do not overlap.
    let staged = unsafe { cx.scratch_down().slice::<f32>(0, f) };
    let out = unsafe { cx.result_0().span_mut([0, rows.start], rows.len()) };
    projection::project_arithmetic(&cx.arg_expert_down(), rows.start, staged, None, out);
    for value in out {
        // Match the CPU routed_down control's unit-weight, zero-base result,
        // including the treatment of signed zero.
        *value = 0.0_f32 + 1.0_f32.mul_add(activation::publish::<Bf16>(*value), 0.0);
    }
}
