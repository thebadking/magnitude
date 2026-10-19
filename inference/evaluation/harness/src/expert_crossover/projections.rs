//! Observe F32 accumulators only after checking the ordinary routed publications.
use super::*;
use seismic::cpu::{Bf16, Dense};

fn values(tensor: &Tensor) -> Result<Vec<f32>, String> {
    require(
        tensor.element() == Element::f32(),
        "projection observation is not F32",
    )?;
    let bytes = tensor.read_to_host().map_err(why)?;
    require(bytes.len() % 4 == 0, "invalid F32 projection bytes")?;
    let result: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte chunk")))
        .collect();
    require(
        result.iter().all(|v| v.is_finite()),
        "nonfinite raw projection",
    )?;
    Ok(result)
}

fn project(
    gpu: &Backend,
    input: &Tensor,
    weight: &Tensor,
    h: u64,
    f: u64,
) -> Result<(Vec<f32>, Value), String> {
    use magnitude_kernels::routed_projection;
    let kernel = prepare(
        routed_projection::native_entry_with(routed_projection::Elements {
            A: Element::bf16(),
            EW: weight.element(),
        }),
        &gpu.device,
        &[("M", 1), ("H", h), ("K", 1), ("F", f)],
    )?;
    require(
        kernel.specialization().launch_param(0, "KSPLIT") == Some(1)
            && kernel.specialization().launch_param(0, "TPW") == Some(1),
        "raw diagnostic requires the one-part, one-tile routed default",
    )?;
    let routes =
        Tensor::from_host(&gpu.device, Element::i32(), &[1, 1], &ints([0])).map_err(why)?;
    let output = kernel
        .call(routed_projection::Args {
            normalized: input,
            routes: &routes,
            expert_weight: weight,
        })
        .map_err(why)?
        .value;
    Ok((values(&output)?, kernel_identity(&kernel)))
}

fn ordered(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits | 0x8000_0000
    } else {
        !bits
    }
}

fn compare(
    cpu: &[f32],
    gpu: &[f32],
    directory: &std::path::Path,
    stem: &str,
) -> Result<Value, String> {
    require(
        !cpu.is_empty() && cpu.len() == gpu.len(),
        "raw projection dimensions differ",
    )?;
    require(
        cpu.iter().chain(gpu).all(|v| v.is_finite()),
        "nonfinite projection comparison",
    )?;
    let (mut f32_equal, mut bf16_equal, mut max_abs, mut max_ulps) = (0, 0, 0.0_f64, 0_u32);
    let (mut squared, mut reference_squared) = (0.0_f64, 0.0_f64);
    let mut samples = Vec::new();
    for (column, (&c, &g)) in cpu.iter().zip(gpu).enumerate() {
        f32_equal += usize::from(c.to_bits() == g.to_bits());
        let ulps = ordered(c).abs_diff(ordered(g));
        max_ulps = max_ulps.max(ulps);
        let delta = f64::from(c) - f64::from(g);
        max_abs = max_abs.max(delta.abs());
        squared += delta * delta;
        reference_squared += f64::from(g) * f64::from(g);
        let (cb, gb) = (Bf16::narrow(c), Bf16::narrow(g));
        bf16_equal += usize::from(cb == gb);
        if cb != gb && samples.len() < 8 {
            samples.push(json!({"column":column,"cpu":c,"gpu":g,
                "cpu_f32_bits":c.to_bits(),"gpu_f32_bits":g.to_bits(),
                "cpu_bf16_bits":cb,"gpu_bf16_bits":gb,"f32_ulps":ulps,
                "cpu_ulps_from_bf16_midpoint":(c.to_bits() & 0xffff).abs_diff(0x8000),
                "gpu_ulps_from_bf16_midpoint":(g.to_bits() & 0xffff).abs_diff(0x8000)}));
        }
    }
    let mut payloads = serde_json::Map::new();
    for (side, data) in [("cpu", cpu), ("gpu", gpu)] {
        let file = format!("{stem}-{side}.f32");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(directory.join(&file), &bytes).map_err(why)?;
        payloads.insert(
            side.into(),
            json!({"file":file,"bytes":bytes.len(),"sha256":sha256_hex(&bytes)}),
        );
    }
    Ok(
        json!({"coordinates":cpu.len(),"equal_f32":f32_equal,"equal_bf16":bf16_equal,
        "max_abs":max_abs,"max_f32_ulps":max_ulps,
        "relative_rms":(squared / reference_squared.max(f64::MIN_POSITIVE)).sqrt(),
        "different_bf16_samples":samples,"raw_f32":payloads}),
    )
}

pub(super) fn observe(
    gpu: &Backend,
    candidate: &grouped::Candidate,
    product: &Tensor,
    contribution: &Tensor,
    activation: &grouped::Activations,
    directory: &std::path::Path,
    stem: &str,
    h: u64,
    f: u64,
) -> Result<Value, String> {
    for specialization in [gpu.up.specialization(), gpu.down.specialization()] {
        require(
            specialization.launch_param(0, "KSPLIT") == Some(1)
                && specialization.launch_param(0, "TPW") == Some(1),
            "raw projection diagnostic does not match the routed parent partition",
        )?;
    }
    let (gate, gate_entry) = project(gpu, &gpu.normalized[0], &gpu.resident[0], h, f)?;
    let (up, up_entry) = project(gpu, &gpu.normalized[0], &gpu.resident[1], h, f)?;
    let rebuilt_product: Vec<u8> = gate
        .iter()
        .zip(&up)
        .flat_map(|(&g, &u)| activation.product(g, u).to_le_bytes())
        .collect();
    require(
        rebuilt_product == product.read_to_host().map_err(why)?,
        "raw projections do not reproduce the routed BF16 gate/up product",
    )?;
    let (down, down_entry) = project(
        gpu,
        &product.reshape(&[1, f]).map_err(why)?,
        &gpu.resident[2],
        f,
        h,
    )?;
    let rebuilt_contribution: Vec<u8> = down
        .iter()
        .flat_map(|&v| (0.0_f32 + 1.0_f32.mul_add(Bf16::round(v), 0.0)).to_le_bytes())
        .collect();
    require(
        rebuilt_contribution == contribution.read_to_host().map_err(why)?,
        "raw projection does not reproduce the routed BF16 down publication",
    )?;
    std::fs::create_dir_all(directory).map_err(why)?;
    Ok(
        json!({"reconstructed_gpu_product_exact":true,"reconstructed_gpu_contribution_exact":true,
        "parent_up":kernel_identity(&gpu.up),"parent_down":kernel_identity(&gpu.down),
        "entries":{"gate":gate_entry,"up":up_entry,"down":down_entry},
        "gate":compare(&candidate.gate,&gate,directory,&format!("{stem}-gate"))?,
        "up":compare(&candidate.up,&up,directory,&format!("{stem}-up"))?,
        "down_same_gpu_product":compare(&candidate.down_same_gpu_product,&down,directory,&format!("{stem}-down"))?,
        "scope":"Verified one-part routed F32 observations; scalar grouped CPU comparison, no serving qualification or timing"}),
    )
}
