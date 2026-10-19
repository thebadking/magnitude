//! A numerical recipe experiment on actual captured operands, not serving.
use super::*;
use seismic::cpu::weights::{
    GroupRounding, Q4K, Q5G32M, Q5G32S, Q5K, Q8, Rows8, affine_grouped_dot,
};
use seismic::cpu::{Bf16, Dense};

pub(super) struct Activations {
    values: Vec<u16>,
    pub diagnostic: Value,
}
impl Activations {
    pub fn product(&self, gate: f32, up: f32) -> u16 {
        Bf16::narrow(Bf16::widen(self.values[usize::from(Bf16::narrow(gate))]) * Bf16::round(up))
    }

    pub fn new(device: &Device) -> Result<Self, String> {
        // Full BF16 function domain, independent of model or captured operands.
        let bytes: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let gate = Tensor::from_host(device, Element::bf16(), &[1, 65536], &bytes).map_err(why)?;
        let one: Vec<u8> = std::iter::repeat_n(Bf16::narrow(1.0), 65536)
            .flat_map(u16::to_le_bytes)
            .collect();
        let up = Tensor::from_host(device, Element::bf16(), &[1, 65536], &one).map_err(why)?;
        let kernel = prepare(
            magnitude_kernels::draft_gated_rows::native_entry_with(
                magnitude_kernels::draft_gated_rows::Elements { A: Element::bf16() },
            ),
            device,
            &[],
        )?;
        let raw = kernel
            .call(magnitude_kernels::draft_gated_rows::Args {
                gate: &gate,
                up: &up,
            })
            .map_err(why)?
            .value
            .read_to_host()
            .map_err(why)?;
        let values: Vec<u16> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut finite_differences = Vec::new();
        for code in 0..=u16::MAX {
            let g = Bf16::widen(code);
            if g.is_finite() && Bf16::narrow(g / (1.0 + (-g).exp())) != values[usize::from(code)] {
                finite_differences.push(code);
            }
        }
        Ok(Self {
            values,
            diagnostic: json!({"domain":"All 65536 BF16 input codes; BF16 SiLU publication",
            "gpu_entry":kernel_identity(&kernel),"gpu_table_sha256":sha256_hex(&raw),
            "finite_cpu_gpu_differences":finite_differences.len(),"different_input_codes":finite_differences,
            "scope":"Pointwise numerical diagnostic, not model quality or admission"}),
        })
    }
}

fn dot(
    weight: &Weight,
    native: &[u8],
    x: &[f32],
    rounding: GroupRounding,
    out: &mut [f32],
) -> Result<(), String> {
    let rows = weight.shape[1] as usize;
    let k = weight.shape[2] as usize;
    let result = match weight.cpu.name() {
        "q4k@rows8" => affine_grouped_dot::<Rows8<Q4K>>(native, rows, k, 0, x, 1, rounding, out),
        "q5k@rows8" => affine_grouped_dot::<Rows8<Q5K>>(native, rows, k, 0, x, 1, rounding, out),
        "q5g32s@rows8" => {
            affine_grouped_dot::<Rows8<Q5G32S>>(native, rows, k, 0, x, 1, rounding, out)
        }
        "q5g32@rows8" => {
            affine_grouped_dot::<Rows8<Q5G32M>>(native, rows, k, 0, x, 1, rounding, out)
        }
        "q8g32s@rows8" => affine_grouped_dot::<Rows8<Q8>>(native, rows, k, 0, x, 1, rounding, out),
        _ => None,
    };
    result.ok_or_else(|| "grouped CPU recipe does not support this weight geometry".into())
}

pub(super) struct Candidate {
    pub product: Tensor,
    pub result: Tensor,
    pub product_gpu_silu: Tensor,
    pub result_gpu_silu: Tensor,
    pub result_same_gpu_product: Tensor,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down_same_gpu_product: Vec<f32>,
}

pub(super) fn evaluate(
    cpu: &Backend,
    weights: &[Weight; 3],
    input: &[u8],
    gpu_product: &Tensor,
    activation: &Activations,
    rounding: GroupRounding,
) -> Result<Candidate, String> {
    let normalized: Vec<f32> = input
        .chunks_exact(2)
        .map(|b| Bf16::widen(u16::from_le_bytes([b[0], b[1]])))
        .collect();
    let native: [Vec<u8>; 3] = triple(
        cpu.resident
            .iter()
            .map(|w| w.read_to_host().map_err(why))
            .collect::<Result<Vec<_>, _>>()?,
    )?;
    let (h, f) = (weights[0].shape[2] as usize, weights[0].shape[1] as usize);
    let (mut gate, mut up) = (vec![0.0_f32; f], vec![0.0_f32; f]);
    dot(&weights[0], &native[0], &normalized, rounding, &mut gate)?;
    dot(&weights[1], &native[1], &normalized, rounding, &mut up)?;
    let mut product = Vec::with_capacity(f);
    let mut gpu_silu = Vec::with_capacity(f);
    for (&g, &u) in gate.iter().zip(&up) {
        let code = Bf16::narrow(g);
        let g = Bf16::widen(code);
        let u = Bf16::round(u);
        product.push(Bf16::narrow(Bf16::round(g / (1.0 + (-g).exp())) * u));
        gpu_silu.push(Bf16::narrow(
            Bf16::widen(activation.values[usize::from(code)]) * u,
        ));
    }
    let actual_gpu_product = gpu_product.read_to_host().map_err(why)?;
    let actual_gpu_product: Vec<u16> = actual_gpu_product
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let down = |product: &[u16]| -> Result<(Tensor, Vec<f32>), String> {
        let x: Vec<f32> = product.iter().map(|&v| Bf16::widen(v)).collect();
        let mut result = vec![0.0_f32; h];
        dot(&weights[2], &native[2], &x, rounding, &mut result)?;
        let bytes: Vec<u8> = result
            .iter()
            .map(|&v| 0.0_f32 + 1.0_f32.mul_add(Bf16::round(v), 0.0))
            .flat_map(f32::to_le_bytes)
            .collect();
        Ok((
            Tensor::from_host(&cpu.device, Element::f32(), &[1, h as u64], &bytes).map_err(why)?,
            result,
        ))
    };
    let (result, _) = down(&product)?;
    let (result_gpu_silu, _) = down(&gpu_silu)?;
    let (result_same_gpu_product, down_same_gpu_product) = down(&actual_gpu_product)?;
    let tensor = |product: Vec<u16>| {
        Tensor::from_host(
            &cpu.device,
            Element::bf16(),
            &[1, 1, f as u64],
            &product
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .map_err(why)
    };
    Ok(Candidate {
        product: tensor(product)?,
        result,
        product_gpu_silu: tensor(gpu_silu)?,
        result_gpu_silu,
        result_same_gpu_product,
        gate,
        up,
        down_same_gpu_product,
    })
}
