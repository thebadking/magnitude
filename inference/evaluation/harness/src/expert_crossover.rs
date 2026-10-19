//! An operator/storage diagnostic. It never dispatches experts for serving.
mod cache;
mod exchange;
mod replay;
use crate::{Inputs, Rendered, digest::sha256_hex, render::RenderedFile};
use clap::Args;
use magnitude_artifacts::gguf::Encoding;
use magnitude_kernels::{repack_weight, routed_bank_fill, routed_down, routed_gate_up};
pub use replay::{ReplayOptions, run as replay};
use seismic::NativeGraphFamily;
use seismic::{
    BackendName, BoundEntry, BoundNativeGraphPlan, Device, DeviceCatalog, DeviceOptions,
    DeviceSelector, Element, Entry, MeasureOptions, NativeGraph, NativeGraphOutputs,
    NativeGraphPlan, NativeGraphSlot, NativeKernel, NativePort, NativeSpecialization, Tensor,
    TraceDetail, TracedSubmission, WorkflowTensor,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fmt::Display, path::PathBuf, sync::Arc, time::Instant};

#[derive(Args)]
pub struct Options {
    /// An own-harness logits export with features, rendered input and result record.
    #[arg(long)]
    feature_capture: PathBuf,
    #[arg(long)]
    pub context: Option<usize>,
    /// Literal GGUF tensor names; the study checks their complete expert geometry.
    #[arg(long)]
    gate_tensor: String,
    #[arg(long)]
    up_tensor: String,
    #[arg(long)]
    down_tensor: String,
    /// An exact DeviceSelector JSON value, not automatic device selection.
    #[arg(long)]
    gpu_selector: String,
    #[arg(long, default_value_t = 32)]
    experts: usize,
    /// The size of the missed subset, one or two experts per activation row.
    #[arg(long, default_value_t = 1)]
    misses: usize,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Check imports and per-backend path equality without collecting timings.
    #[arg(long)]
    qualify_only: bool,
    /// Qualify the physical CPU-ready cache with real weights and fixture vectors.
    #[arg(long)]
    cpu_cache_prototype: bool,
    /// Qualify the parked CUDA/CPU activation and contribution protocol.
    #[arg(long)]
    cpu_exchange_prototype: bool,
    /// Exact second CUDA selector for the dual-owner exchange qualification.
    #[arg(long)]
    cpu_exchange_peer_selector: Option<String>,
}

fn why(error: impl Display) -> String {
    error.to_string()
}
fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn triple<T>(values: Vec<T>) -> Result<[T; 3], String> {
    values
        .try_into()
        .map_err(|_| "expected a gate/up/down triple".into())
}
fn ints(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
fn prepare<E: Entry>(
    entry: BoundEntry<E>,
    device: &Device,
    dimensions: &[(&str, u64)],
) -> Result<NativeKernel<E>, String> {
    let mut statics = NativeSpecialization::new();
    for &(name, value) in dimensions {
        statics = statics.with_static(name, value);
    }
    let default = entry
        .default_specialization(device, &statics)
        .map_err(why)?
        .ok_or_else(|| format!("{} has no canonical implementation", E::NAME))?;
    entry.prepare(device, &default).map_err(why)
}

struct Weight {
    shape: Vec<u64>,
    raw: Vec<u8>,
    source: Element,
    cpu: Element,
    gpu: Element,
    record_bytes: usize,
    identity: Value,
}

fn representations(encoding: Encoding) -> Result<(Element, Element, Element), String> {
    let (source, resident) = match encoding {
        Encoding::Q4K => ("gguf_q4_k", "q4k"),
        Encoding::Q5K => ("gguf_q5_k", "q5k"),
        Encoding::Q5_0 => ("gguf_q5_0", "q5g32s"),
        Encoding::Q5_1 => ("gguf_q5_1", "q5g32"),
        Encoding::Q8_0 => ("gguf_q8_0", "q8g32s"),
        _ => return Err(format!("unsupported study encoding {encoding:?}")),
    };
    let element =
        |name: &str| Element::named(name).ok_or_else(|| format!("unknown element {name}"));
    Ok((
        element(source)?,
        element(&format!("{resident}@rows8"))?,
        element(&format!("{resident}@mma16"))?,
    ))
}

fn capture(
    inputs: &Inputs,
    options: &Options,
    rendered: &[Rendered],
    digest: &str,
) -> Result<(Vec<Vec<u8>>, Value), String> {
    let root = &options.feature_capture;
    let record_bytes = std::fs::read(root.join("result.json")).map_err(why)?;
    let record: Value = serde_json::from_slice(&record_bytes).map_err(why)?;
    require(
        record["tool"] == "harness.logits",
        "features must come from the own logits harness",
    )?;
    require(
        record["model"] == inputs.artifacts.artifacts.model,
        "capture model differs",
    )?;
    require(
        record["artifacts"]["target_sha256"] == inputs.artifacts.artifacts.target_sha256,
        "capture artifact differs",
    )?;
    require(
        record["rendered_digest"] == digest,
        "capture rendered input differs",
    )?;
    let captured = RenderedFile::read(
        &root.join("rendered.jsonl"),
        &inputs.set.digest,
        &inputs.artifacts.artifacts.model,
        &inputs.artifacts.artifacts.target_sha256,
    )?;
    require(captured.digest == digest, "capture rendered file differs")?;
    let metadata_bytes = std::fs::read(root.join("features.json")).map_err(why)?;
    let metadata: Value = serde_json::from_slice(&metadata_bytes).map_err(why)?;
    require(
        metadata["rendered_digest"] == digest && metadata["element"] == "bf16",
        "capture feature representation or input identity differs",
    )?;
    let width = metadata["width"].as_u64().ok_or("missing feature width")?;
    let row_bytes = metadata["row_bytes"]
        .as_u64()
        .ok_or("missing feature row bytes")? as usize;
    require(
        row_bytes as u64 == width.checked_mul(2).ok_or("feature width overflow")?,
        "invalid BF16 row size",
    )?;
    let rows = metadata["rows"]
        .as_array()
        .ok_or("missing feature row mapping")?;
    require(
        !rows.is_empty() && rows.len() <= 128,
        "bounded feature capture needs 1..128 rows",
    )?;
    let bytes = std::fs::read(root.join("features.bin")).map_err(why)?;
    require(
        bytes.len()
            == rows
                .len()
                .checked_mul(row_bytes)
                .ok_or("capture byte overflow")?,
        "capture length differs",
    )?;
    for row in rows {
        let id = row["fixture_id"]
            .as_str()
            .ok_or("missing feature fixture")?;
        let position = row["position"].as_u64().ok_or("missing feature position")? as usize;
        require(
            rendered.iter().any(|fixture| {
                fixture.fixture_id == id
                    && position < fixture.tokens.len()
                    && fixture
                        .scored_spans
                        .iter()
                        .any(|span| span.contains(&(position + 1)))
            }),
            "feature row is outside the fixture's scored positions",
        )?;
    }
    let values = Element::bf16()
        .decode_host(&[rows.len() as u64, width], &bytes)
        .ok_or("cannot decode captured features")?;
    require(
        values.iter().all(|x| x.is_finite()) && values.iter().any(|&x| x != 0.0),
        "invalid captured feature values",
    )?;
    Ok((
        bytes.chunks_exact(row_bytes).map(<[u8]>::to_vec).collect(),
        json!({
            "directory": root, "producer_build": record["build"], "rendered_digest": digest,
            "features_sha256": sha256_hex(&bytes), "metadata_sha256": sha256_hex(&metadata_bytes),
            "producer_record_sha256": sha256_hex(&record_bytes), "metadata": metadata,
            "use": "Readout features used as expert projection operands; not captured routed-layer activations"
        }),
    ))
}

struct Backend {
    device: Device,
    raw: [Tensor; 3],
    resident: [Tensor; 3],
    imports: [NativeKernel<repack_weight::Entry>; 3],
    up: NativeKernel<routed_gate_up::Entry>,
    down: NativeKernel<routed_down::Entry>,
    fill: Option<NativeKernel<routed_bank_fill::Entry>>,
    normalized: Vec<Tensor>,
}

impl Backend {
    fn new(
        device: Device,
        weights: &[Weight; 3],
        features: &[Vec<u8>],
        k: u64,
    ) -> Result<Self, String> {
        let cuda = device.backend() == BackendName::Cuda;
        let reps = weights
            .each_ref()
            .map(|weight| if cuda { weight.gpu } else { weight.cpu });
        let mut raw = Vec::new();
        let mut imports = Vec::new();
        let mut resident = Vec::new();
        for kind in 0..3 {
            raw.push(
                Tensor::from_host(
                    &device,
                    weights[kind].source,
                    &weights[kind].shape,
                    &weights[kind].raw,
                )
                .map_err(why)?,
            );
            let import = prepare(
                repack_weight::native_entry_with(repack_weight::Elements {
                    E: weights[kind].source,
                    U: reps[kind],
                }),
                &device,
                &[],
            )?;
            resident.push(
                import
                    .call(repack_weight::Args { source: &raw[kind] })
                    .map_err(why)?
                    .value,
            );
            let shape = [1, weights[kind].shape[1], weights[kind].shape[2]];
            let expected = reps[kind]
                .repack_host(
                    weights[kind].source,
                    &shape,
                    &weights[kind].raw[..weights[kind].record_bytes],
                )
                .ok_or("no registered host packing reference")?;
            let observed = resident[kind]
                .slice_leading(0, 1)
                .map_err(why)?
                .read_to_host()
                .map_err(why)?;
            require(
                observed == expected,
                "native import differs from the registered packing reference",
            )?;
            imports.push(import);
        }
        let h = weights[0].shape[2];
        let f = weights[0].shape[1];
        let up_statics = if cuda {
            vec![("H", h), ("K", k), ("F", f)]
        } else {
            vec![]
        };
        let down_statics = if cuda {
            vec![("H", h), ("K", 1), ("F", f)]
        } else {
            vec![]
        };
        let up = prepare(
            routed_gate_up::native_entry_with(routed_gate_up::Elements {
                A: Element::bf16(),
                EGW: reps[0],
                EUW: reps[1],
            }),
            &device,
            &up_statics,
        )?;
        let down = prepare(
            routed_down::native_entry_with(routed_down::Elements {
                A: Element::bf16(),
                R: Element::f32(),
                EDW: reps[2],
            }),
            &device,
            &down_statics,
        )?;
        let fill = if cuda {
            Some(prepare(
                routed_bank_fill::native_entry_with(routed_bank_fill::Elements {
                    SG: weights[0].source,
                    SU: weights[1].source,
                    SD: weights[2].source,
                    EGW: reps[0],
                    EUW: reps[1],
                    EDW: reps[2],
                }),
                &device,
                &[
                    ("H", h),
                    ("K", k),
                    ("F", f),
                    ("T", weights[0].shape[0]),
                    ("C", 4 + 8 * k),
                    ("P", 8 * k),
                ],
            )?)
        } else {
            None
        };
        let normalized = features
            .iter()
            .map(|bytes| Tensor::from_host(&device, Element::bf16(), &[1, h], bytes).map_err(why))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            device,
            raw: triple(raw)?,
            resident: triple(resident)?,
            imports: triple(imports)?,
            up,
            down,
            fill,
            normalized,
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    CpuResident,
    CpuImport,
    GpuResident,
    GpuMiss,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::CpuResident => "cpu_resident",
            Self::CpuImport => "cpu_import",
            Self::GpuResident => "gpu_resident",
            Self::GpuMiss => "gpu_miss",
        }
    }
    fn cpu(self) -> bool {
        matches!(self, Self::CpuResident | Self::CpuImport)
    }
}

fn port(
    graph: &mut NativeGraph,
    ports: &mut Vec<(String, NativePort)>,
    name: &str,
    element: Element,
    shape: &[u64],
) -> Result<WorkflowTensor, String> {
    let value = graph.port(element, shape).map_err(why)?;
    let tensor = value.tensor().clone();
    ports.push((name.to_owned(), value));
    Ok(tensor)
}

struct Circuit {
    plan: NativeGraphPlan,
    ports: Vec<(String, NativePort)>,
    slot: NativeGraphSlot,
    outputs: Option<NativeGraphOutputs>,
    product: WorkflowTensor,
    result: WorkflowTensor,
}
struct Output {
    product: Tensor,
    result: Tensor,
}

impl Circuit {
    fn new(data: &Backend, weights: &[Weight; 3], mode: Mode, k: u64) -> Result<Self, String> {
        let mut graph = data.device.native_graph();
        let mut ports = Vec::new();
        let h = weights[0].shape[2];
        let f = weights[0].shape[1];
        let count = weights[0].shape[0];
        let e = if mode == Mode::CpuImport { k } else { count };
        let norm = port(
            &mut graph,
            &mut ports,
            "normalized",
            Element::bf16(),
            &[1, h],
        )?;
        let base = port(&mut graph, &mut ports, "base", Element::f32(), &[k, h])?;
        let coefficients = port(
            &mut graph,
            &mut ports,
            "coefficients",
            Element::f32(),
            &[k, 1],
        )?;
        let mut raw = Vec::new();
        if matches!(mode, Mode::CpuImport | Mode::GpuMiss) {
            for kind in 0..3 {
                raw.push(port(
                    &mut graph,
                    &mut ports,
                    &format!("raw{kind}"),
                    weights[kind].source,
                    &[e, weights[kind].shape[1], weights[kind].shape[2]],
                )?);
            }
        }
        let mut resident = Vec::new();
        if mode == Mode::CpuImport {
            for kind in 0..3 {
                resident.push(
                    graph
                        .enqueue(
                            &data.imports[kind],
                            repack_weight::WorkflowArgs {
                                source: (&raw[kind]).into(),
                            },
                        )
                        .map_err(why)?
                        .value,
                );
            }
        } else {
            for kind in 0..3 {
                resident.push(port(
                    &mut graph,
                    &mut ports,
                    &format!("resident{kind}"),
                    data.resident[kind].element(),
                    &[e, weights[kind].shape[1], weights[kind].shape[2]],
                )?);
            }
        }
        let routes = if mode == Mode::GpuMiss {
            let slots = port(&mut graph, &mut ports, "slots", Element::i32(), &[1, k])?;
            let mut stamps = port(&mut graph, &mut ports, "stamps", Element::i32(), &[count])?;
            let step = port(&mut graph, &mut ports, "step", Element::i32(), &[1])?;
            let mut control = port(
                &mut graph,
                &mut ports,
                "control",
                Element::i32(),
                &[4 + 8 * k],
            )?;
            let placement = port(
                &mut graph,
                &mut ports,
                "placement",
                Element::i32(),
                &[5, 8 * k],
            )?;
            let heartbeat = port(&mut graph, &mut ports, "heartbeat", Element::i32(), &[1])?;
            let mut slot_map = port(&mut graph, &mut ports, "slot_map", Element::i32(), &[count])?;
            let mut commits = port(&mut graph, &mut ports, "commits", Element::i32(), &[count])?;
            let mut gate = resident[0].clone();
            let mut up = resident[1].clone();
            let mut down = resident[2].clone();
            graph
                .enqueue(
                    data.fill.as_ref().ok_or("missing CUDA bank converter")?,
                    routed_bank_fill::WorkflowArgs {
                        slots: (&slots).into(),
                        stamps: (&mut stamps).into(),
                        step: (&step).into(),
                        control: (&mut control).into(),
                        placement: (&placement).into(),
                        heartbeat: (&heartbeat).into(),
                        staged_gate: (&raw[0]).into(),
                        staged_up: (&raw[1]).into(),
                        staged_down: (&raw[2]).into(),
                        expert_gate: (&mut gate).into(),
                        expert_up: (&mut up).into(),
                        expert_down: (&mut down).into(),
                        slot_map: (&mut slot_map).into(),
                        commits: (&mut commits).into(),
                    },
                )
                .map_err(why)?
                .value
        } else {
            port(&mut graph, &mut ports, "routes", Element::i32(), &[1, k])?
        };
        let product = graph
            .enqueue(
                &data.up,
                routed_gate_up::WorkflowArgs {
                    normalized: (&norm).into(),
                    routes: (&routes).into(),
                    expert_gate: (&resident[0]).into(),
                    expert_up: (&resident[1]).into(),
                    activation: 0,
                },
            )
            .map_err(why)?
            .value;
        let product_rows = product.reshape(&[k, 1, f]);
        let route_rows = routes.reshape(&[k, 1]);
        let result = graph
            .enqueue(
                &data.down,
                routed_down::WorkflowArgs {
                    base: (&base).into(),
                    product: (&product_rows).into(),
                    routes: (&route_rows).into(),
                    weights: (&coefficients).into(),
                    expert_down: (&resident[2]).into(),
                },
            )
            .map_err(why)?
            .value;
        graph.export(&product).map_err(why)?;
        graph.export(&result).map_err(why)?;
        let plan = graph.seal().map_err(why)?;
        let slot = plan.new_slot().map_err(why)?;
        let family = NativeGraphFamily::new(std::slice::from_ref(&plan)).map_err(why)?;
        let outputs = Some(
            family
                .new_output_slot()
                .map_err(why)?
                .activate(&plan)
                .map_err(why)?,
        );
        Ok(Self {
            plan,
            ports,
            slot,
            outputs,
            product,
            result,
        })
    }
    fn bind(&self, values: &BTreeMap<String, Tensor>) -> Result<BoundNativeGraphPlan, String> {
        let pairs = self
            .ports
            .iter()
            .map(|(name, port)| {
                values
                    .get(name)
                    .map(|tensor| (port, tensor))
                    .ok_or_else(|| format!("missing binding {name}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.plan.bind_static(&pairs).map_err(why)
    }
    fn execute(&mut self, bound: &BoundNativeGraphPlan) -> Result<Output, String> {
        let outputs = self
            .outputs
            .take()
            .ok_or("missing output lease")?
            .recycle()
            .map_err(why)?
            .activate(&self.plan)
            .map_err(why)?;
        let (outputs, completion) = self
            .slot
            .attach(bound.bindings(), outputs)
            .map_err(why)?
            .submit()
            .map_err(why)?;
        completion.wait().map_err(why)?;
        let result = outputs
            .exported(&self.result)
            .ok_or("missing projection output")?;
        let product = outputs
            .exported(&self.product)
            .ok_or("missing product output")?;
        self.outputs = Some(outputs);
        Ok(Output { product, result })
    }
}

fn bindings(
    data: &Backend,
    mode: Mode,
    weights: &[Weight; 3],
    first: usize,
    k: usize,
) -> Result<BTreeMap<String, Tensor>, String> {
    let mut values = BTreeMap::new();
    let h = weights[0].shape[2];
    values.insert(
        "normalized".into(),
        data.normalized[first % data.normalized.len()].clone(),
    );
    values.insert(
        "base".into(),
        Tensor::zeros(&data.device, Element::f32(), &[k as u64, h]).map_err(why)?,
    );
    let ones = vec![1_f32.to_le_bytes(); k]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    values.insert(
        "coefficients".into(),
        Tensor::from_host(&data.device, Element::f32(), &[k as u64, 1], &ones).map_err(why)?,
    );
    if mode == Mode::CpuImport {
        for kind in 0..3 {
            values.insert(
                format!("raw{kind}"),
                data.raw[kind]
                    .slice_leading(first as u64, (first + k) as u64)
                    .map_err(why)?,
            );
        }
    } else {
        for kind in 0..3 {
            values.insert(format!("resident{kind}"), data.resident[kind].clone());
        }
        if mode == Mode::GpuMiss {
            for kind in 0..3 {
                values.insert(format!("raw{kind}"), data.raw[kind].clone());
            }
        }
    }
    let count = weights[0].shape[0];
    if mode == Mode::GpuMiss {
        let p = 8 * k;
        let mut control = vec![0; 4 + 8 * k];
        control[0] = 1;
        control[1] = 1;
        control[3] = k as i32;
        let mut placement = vec![0; 5 * p];
        for c in 0..k {
            control[4 + c] = (first + c) as i32;
            for row in 0..4 {
                placement[row * p + c] = (first + c) as i32;
            }
            placement[4 * p + c] = 1;
        }
        for (name, shape, words) in [
            ("slots", vec![1, k as u64], vec![-1; k]),
            ("stamps", vec![count], vec![0; count as usize]),
            ("step", vec![1], vec![1]),
            ("heartbeat", vec![1], vec![1]),
            ("control", vec![(4 + 8 * k) as u64], control),
            ("placement", vec![5, p as u64], placement),
            ("slot_map", vec![count], vec![-1; count as usize]),
            ("commits", vec![count], vec![0; count as usize]),
        ] {
            let bytes = ints(words);
            let tensor = if matches!(name, "slots" | "step") {
                Tensor::from_host(&data.device, Element::i32(), &shape, &bytes).map_err(why)?
            } else {
                // Match the streaming bank's mapped host directory. The
                // acknowledged control state is immutable during each run;
                // mutable directory words are only accessed after completion.
                let tensor = Tensor::shared(&data.device, Element::i32(), &shape).map_err(why)?;
                unsafe { tensor.write_shared(0, &bytes) }.map_err(why)?;
                tensor
            };
            values.insert(name.into(), tensor);
        }
    } else {
        let start = if mode == Mode::CpuImport { 0 } else { first };
        values.insert(
            "routes".into(),
            Tensor::from_host(
                &data.device,
                Element::i32(),
                &[1, k as u64],
                &ints(start as i32..(start + k) as i32),
            )
            .map_err(why)?,
        );
    }
    Ok(values)
}

fn upload(
    gpu: &Backend,
    pinned: &[Tensor; 3],
    weights: &[Weight; 3],
    first: usize,
    k: usize,
) -> Result<(), String> {
    for kind in 0..3 {
        let offset = first * weights[kind].record_bytes;
        let bytes = k * weights[kind].record_bytes;
        let (pointer, length) = pinned[kind].host_window().map_err(why)?;
        require(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= length as usize),
            "pinned upload range overflow",
        )?;
        // Pinned source tensors are initialized once and never written by device work.
        // All readers of the destination finished; no new graph is queued until
        // the protocol upload completes. Both owners outlive the synchronous copy.
        let source = unsafe { std::slice::from_raw_parts(pointer.as_ptr().add(offset), bytes) };
        unsafe { gpu.raw[kind].write_in_protocol(offset as u64, source) }.map_err(why)?;
    }
    Ok(())
}

fn difference(left: &Tensor, right: &Tensor) -> Result<Value, String> {
    let a = left.read_to_host().map_err(why)?;
    let b = right.read_to_host().map_err(why)?;
    require(
        left.element() == right.element() && left.extents() == right.extents(),
        "comparison shape differs",
    )?;
    let av = left
        .element()
        .decode_host(left.extents(), &a)
        .ok_or("cannot decode comparison output")?;
    let bv = right
        .element()
        .decode_host(right.extents(), &b)
        .ok_or("cannot decode comparison output")?;
    let mut squared = 0.0_f64;
    let mut reference = 0.0_f64;
    let mut maximum = 0.0_f64;
    for (&x, &y) in av.iter().zip(&bv) {
        require(x.is_finite() && y.is_finite(), "nonfinite expert output")?;
        squared += (x - y).powi(2);
        reference += x * x;
        maximum = maximum.max((x - y).abs());
    }
    Ok(
        json!({"elements": av.len(), "byte_equal": a == b, "max_abs": maximum,
        "rms_error": (squared / av.len() as f64).sqrt(),
        "relative_rms": if reference > 0.0 { Some((squared / reference).sqrt()) } else { None },
        "diagnostic_only": true}),
    )
}

fn load_weights(inputs: &Inputs, options: &Options) -> Result<[Weight; 3], String> {
    let host = crate::host(&inputs.artifacts)?;
    let mut weights = Vec::new();
    let mut total = 0_u64;
    for name in [
        &options.gate_tensor,
        &options.up_tensor,
        &options.down_tensor,
    ] {
        let stored = host.package().target().tensor(name).map_err(why)?;
        require(
            stored.shape.len() == 3 && options.experts as u64 <= stored.shape[0],
            "weight must be an expert-major three-dimensional tensor",
        )?;
        let (source, cpu, gpu) = representations(stored.encoding)?;
        let one = [1, stored.shape[1], stored.shape[2]];
        let record_bytes = source.canonical_byte_len(&one).map_err(why)?;
        require(
            record_bytes.checked_mul(stored.shape[0]) == Some(stored.nbytes),
            "expert record size does not match the container",
        )?;
        let mut shape = stored.shape.clone();
        shape[0] = options.experts as u64;
        let bytes = record_bytes
            .checked_mul(shape[0])
            .ok_or("weight size overflow")?;
        total = total.checked_add(bytes).ok_or("weight set size overflow")?;
        require(total <= 256 << 20, "study raw weight set exceeds 256 MiB")?;
        let raw = stored
            .source
            .read(stored.offset, bytes as usize)
            .map_err(why)?;
        let identity = json!({"tensor": name, "encoding": stored.encoding, "stored_shape": stored.shape,
            "offset": stored.offset, "selected_shape": shape, "selected_bytes": bytes,
            "selected_sha256": sha256_hex(&raw), "record_bytes": record_bytes,
            "cpu_representation": format!("{cpu:?}"), "gpu_representation": format!("{gpu:?}")});
        weights.push(Weight {
            shape,
            raw,
            source,
            cpu,
            gpu,
            record_bytes: record_bytes as usize,
            identity,
        });
    }
    let weights = triple(weights)?;
    let [e, f, h]: [u64; 3] = weights[0]
        .shape
        .clone()
        .try_into()
        .map_err(|_| "gate geometry")?;
    require(
        weights[1].shape == [e, f, h] && weights[2].shape == [e, h, f],
        "gate/up/down geometry differs",
    )?;
    // This is a working-set guard, not a claim that the two CCD caches form a
    // single shared cache. The selected raw and native sets each exceed both.
    require(
        total > 64 << 20,
        "rotation must exceed this host's combined 64 MiB L3",
    )?;
    Ok(weights)
}

struct Case {
    first: usize,
    values: [BTreeMap<String, Tensor>; 4],
    bound: [BoundNativeGraphPlan; 4],
    returned: Tensor,
}

const MODES: [Mode; 4] = [
    Mode::CpuResident,
    Mode::CpuImport,
    Mode::GpuResident,
    Mode::GpuMiss,
];

fn execute(
    mode: usize,
    case: &Case,
    circuits: &mut [Circuit; 4],
    gpu: &Backend,
    pinned: &[Tensor; 3],
    weights: &[Weight; 3],
    k: usize,
) -> Result<(Output, Value), String> {
    let mode_value = MODES[mode];
    let start = Instant::now();
    let mut activation_seconds = 0.0;
    let mut upload_seconds = 0.0;
    let mut return_seconds = 0.0;
    if mode_value.cpu() {
        let transfer = Instant::now();
        let activation = gpu.normalized[case.first % gpu.normalized.len()]
            .read_to_host()
            .map_err(why)?;
        let mut normalized = case.values[mode]["normalized"].clone();
        normalized.write_from_host(&activation).map_err(why)?;
        activation_seconds = transfer.elapsed().as_secs_f64();
    } else if mode_value == Mode::GpuMiss {
        let transfer = Instant::now();
        upload(gpu, pinned, weights, case.first, k)?;
        upload_seconds = transfer.elapsed().as_secs_f64();
    }
    let graph_start = Instant::now();
    let output = circuits[mode].execute(&case.bound[mode])?;
    let graph_seconds = graph_start.elapsed().as_secs_f64();
    if mode_value.cpu() {
        let transfer = Instant::now();
        // Native CPU output storage has no public protocol host window. Its
        // canonical host readback/staging copy is part of transport here.
        let source = output.result.read_to_host().map_err(why)?;
        // The graph completed and returned has no queued consumers. This
        // staging allocation outlives the synchronous CUDA upload.
        unsafe { case.returned.write_in_protocol(0, &source) }.map_err(why)?;
        return_seconds = transfer.elapsed().as_secs_f64();
    }
    let pipeline_seconds = start.elapsed().as_secs_f64();
    Ok((
        output,
        json!({"pipeline_seconds": pipeline_seconds, "graph_host_seconds": graph_seconds,
        "activation_transport_seconds": activation_seconds, "weight_upload_seconds": upload_seconds,
        "result_transport_seconds": return_seconds}),
    ))
}

fn trace_record(records: Vec<TracedSubmission>, mode: Mode) -> Result<Value, String> {
    require(!records.is_empty(), "operator trace is empty")?;
    let mut entries = BTreeMap::<String, usize>::new();
    let mut seconds = 0.0;
    let mut spans = Vec::new();
    for record in &records {
        require(
            record.device.0.is_finite()
                && record.device.1.is_finite()
                && record.device.1 >= record.device.0,
            "invalid operator device interval",
        )?;
        seconds += record.device.1 - record.device.0;
        for launch in &record.launches {
            if launch.geometry.is_some() {
                *entries.entry(launch.entry.clone()).or_default() += 1;
            }
        }
        spans.push(
            json!({"encode_start": record.encode_start, "committed": record.committed,
            "device": [record.device.0, record.device.1]}),
        );
    }
    let expected = [
        (
            <routed_gate_up::Entry as Entry>::NAME,
            if mode.cpu() { 2 } else { 1 },
        ),
        (
            <routed_down::Entry as Entry>::NAME,
            if mode.cpu() { 2 } else { 1 },
        ),
        (
            <repack_weight::Entry as Entry>::NAME,
            if mode == Mode::CpuImport { 3 } else { 0 },
        ),
        (
            <routed_bank_fill::Entry as Entry>::NAME,
            if mode == Mode::GpuMiss { 1 } else { 0 },
        ),
    ];
    for (entry, count) in expected {
        require(
            entries.get(entry).copied().unwrap_or(0) == count,
            &format!("incomplete or unexpected {entry} trace: {entries:?}"),
        )?;
    }
    Ok(
        json!({"device_interval_sum_seconds": seconds, "submission_count": records.len(),
        "entries": entries, "submissions": spans}),
    )
}

fn kernel_identity<E: Entry>(kernel: &NativeKernel<E>) -> Value {
    json!({"entry": E::NAME, "specialization": format!("{:?}", kernel.specialization()),
        "artifact": format!("{:?}", kernel.artifact())})
}

fn backend_identity(backend: &Backend) -> Value {
    json!({"name": backend.device.info().name, "selector": backend.device.info().selector,
        "up": kernel_identity(&backend.up), "down": kernel_identity(&backend.down),
        "imports": backend.imports.each_ref().map(kernel_identity),
        "fill": backend.fill.as_ref().map(kernel_identity)})
}

pub fn run(
    inputs: &Inputs,
    options: &Options,
    rendered: &[Rendered],
    digest: &str,
) -> Result<(Value, Vec<String>), String> {
    require(
        !(options.cpu_cache_prototype || options.cpu_exchange_prototype)
            || (options.qualify_only && options.misses == 1),
        "CPU prototypes require --qualify-only and --misses 1",
    )?;
    require(
        options.cpu_exchange_prototype == options.cpu_exchange_peer_selector.is_some(),
        "CPU exchange qualification requires its explicit second GPU selector",
    )?;
    require(
        options.experts >= 32
            && options.experts <= 64
            && matches!(options.misses, 1 | 2)
            && options.experts % options.misses == 0,
        "study requires 32..64 experts and one or two misses",
    )?;
    let (features, provenance) = capture(inputs, options, rendered, digest)?;
    let weights = load_weights(inputs, options)?;
    require(
        features
            .iter()
            .all(|row| row.len() as u64 == weights[0].shape[2] * 2),
        "feature width differs from expert input",
    )?;
    let selector: DeviceSelector = serde_json::from_str(&options.gpu_selector).map_err(why)?;
    require(
        matches!(selector, DeviceSelector::Cuda { .. }),
        "study needs one explicit CUDA selector",
    )?;
    let artifacts: Option<Arc<dyn seismic::ArtifactStore>> = options
        .cache_dir
        .as_ref()
        .map(|path| {
            magnitude_executor::KernelCache::open(
                path.clone(),
                magnitude_executor::DEFAULT_KERNEL_CACHE_BYTES,
            )
            .map(|cache| Arc::new(cache) as Arc<dyn seismic::ArtifactStore>)
            .map_err(why)
        })
        .transpose()?;
    let catalog = DeviceCatalog::discover().map_err(why)?;
    let device_options = DeviceOptions {
        artifacts,
        ..DeviceOptions::default()
    };
    let cpu_device = catalog
        .open_with(
            catalog.resolve(DeviceSelector::HostCpu).map_err(why)?,
            device_options.clone(),
        )
        .map_err(why)?;
    let gpu_device = catalog
        .open_with(
            catalog.resolve(selector).map_err(why)?,
            device_options.clone(),
        )
        .map_err(why)?;
    cpu_device.set_memory_limit(Some(1 << 30));
    gpu_device.set_memory_limit(Some(1 << 30));
    let cpu = Backend::new(cpu_device, &weights, &features, options.misses as u64)?;
    let gpu = Backend::new(gpu_device, &weights, &features, options.misses as u64)?;
    let pinned = triple(
        weights
            .iter()
            .map(|weight| {
                let tensor =
                    Tensor::shared(&gpu.device, weight.source, &weight.shape).map_err(why)?;
                // No device work references this mapped host allocation during setup.
                unsafe { tensor.write_shared(0, &weight.raw) }.map_err(why)?;
                Ok(tensor)
            })
            .collect::<Result<Vec<_>, String>>()?,
    )?;
    let mut circuits: [Circuit; 4] = MODES
        .map(|mode| {
            Circuit::new(
                if mode.cpu() { &cpu } else { &gpu },
                &weights,
                mode,
                options.misses as u64,
            )
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "circuit count")?;
    let mut cases = Vec::new();
    for first in (0..options.experts).step_by(options.misses) {
        let values: [BTreeMap<String, Tensor>; 4] = MODES
            .map(|mode| {
                bindings(
                    if mode.cpu() { &cpu } else { &gpu },
                    mode,
                    &weights,
                    first,
                    options.misses,
                )
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "binding count")?;
        let bound = (0..4)
            .map(|i| circuits[i].bind(&values[i]))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "bound count")?;
        let returned = Tensor::zeros(
            &gpu.device,
            Element::f32(),
            &[options.misses as u64, weights[0].shape[2]],
        )
        .map_err(why)?;
        cases.push(Case {
            first,
            values,
            bound,
            returned,
        });
    }
    let mut qualification = Vec::new();
    for case in &cases {
        let mut observed = Vec::new();
        // Each circuit has its own output lease. Read them before the next
        // case reuses those four leases.
        for mode in 0..4 {
            let (output, _) = execute(
                mode,
                case,
                &mut circuits,
                &gpu,
                &pinned,
                &weights,
                options.misses,
            )?;
            observed.push(output);
        }
        let cpu_product = difference(&observed[0].product, &observed[1].product)?;
        let cpu_result = difference(&observed[0].result, &observed[1].result)?;
        let gpu_product = difference(&observed[2].product, &observed[3].product)?;
        let gpu_result = difference(&observed[2].result, &observed[3].result)?;
        let equal = [&cpu_product, &cpu_result, &gpu_product, &gpu_result]
            .iter()
            .all(|d| d["byte_equal"] == true);
        qualification.push(
            json!({"first_expert": case.first, "cpu_product": cpu_product, "cpu_result": cpu_result,
            "gpu_product": gpu_product, "gpu_result": gpu_result,
            "cross_backend_product": difference(&observed[0].product, &observed[2].product)?,
            "cross_backend_result": difference(&observed[0].result, &observed[2].result)?}),
        );
        std::fs::write(
            inputs.run.path("qualification.json"),
            serde_json::to_vec_pretty(&qualification).map_err(why)?,
        )
        .map_err(why)?;
        if !equal {
            for (mode, output) in observed.iter().enumerate() {
                for (name, tensor) in [("product", &output.product), ("result", &output.result)] {
                    std::fs::write(
                        inputs.run.path(&format!(
                            "failed-{}-{}-{name}.bin",
                            case.first,
                            MODES[mode].name()
                        )),
                        tensor.read_to_host().map_err(why)?,
                    )
                    .map_err(why)?;
                }
            }
            return Err(format!(
                "resident/import path equality failed: {}",
                qualification.last().unwrap()
            ));
        }
    }
    let cache_qualification = if options.cpu_cache_prototype {
        let result = cache::qualify(&cpu, &weights, &catalog)?;
        std::fs::write(
            inputs.run.path("cpu-cache-qualification.json"),
            serde_json::to_vec_pretty(&result).map_err(why)?,
        )
        .map_err(why)?;
        Some(result)
    } else {
        None
    };
    let exchange_qualification = if options.cpu_exchange_prototype {
        let result = exchange::qualify(
            &cpu,
            &gpu,
            &weights,
            &catalog,
            options.cpu_exchange_peer_selector.as_deref().unwrap(),
            device_options,
        )?;
        std::fs::write(
            inputs.run.path("cpu-exchange-qualification.json"),
            serde_json::to_vec_pretty(&result).map_err(why)?,
        )
        .map_err(why)?;
        Some(result)
    } else {
        None
    };
    let mut rounds = Vec::new();
    if !options.qualify_only {
        let cpu_trace = cpu
            .device
            .trace_submissions(TraceDetail::Submissions)
            .map_err(why)?;
        let gpu_trace = gpu
            .device
            .trace_submissions(TraceDetail::Submissions)
            .map_err(why)?;
        // GPU/CPU/CPU/GPU for resident operators and both crossovers. Each pass
        // rotates the entire selected set; correctness/setup/trace collection and
        // sustained-clock conditioning are outside the pipeline interval.
        for (pair, sequence) in [
            ("resident_operators", [2, 0, 0, 2]),
            ("resident_cpu_vs_gpu_miss", [3, 0, 0, 3]),
            ("cpu_import_vs_gpu_miss", [3, 1, 1, 3]),
        ] {
            for (pass, mode) in sequence.into_iter().enumerate() {
                let rotation = cases
                    .iter()
                    .map(|case| routed_gate_up::Args {
                        normalized: &case.values[2]["normalized"],
                        routes: &case.values[2]["routes"],
                        expert_gate: &gpu.resident[0],
                        expert_up: &gpu.resident[1],
                        activation: 0,
                    })
                    .collect();
                let conditioning = gpu
                    .up
                    .measure(
                        rotation,
                        &MeasureOptions {
                            samples: 1,
                            min_sample_seconds: 0.002,
                        },
                    )
                    .map_err(why)?;
                for case in &cases {
                    drop(execute(
                        mode,
                        case,
                        &mut circuits,
                        &gpu,
                        &pinned,
                        &weights,
                        options.misses,
                    )?);
                }
                cpu_trace.collect().map_err(why)?;
                gpu_trace.collect().map_err(why)?;
                let mut samples = Vec::new();
                for case in &cases {
                    let (output, mut sample) = execute(
                        mode,
                        case,
                        &mut circuits,
                        &gpu,
                        &pinned,
                        &weights,
                        options.misses,
                    )?;
                    drop(output);
                    let cpu_records = cpu_trace.collect().map_err(why)?;
                    let gpu_records = gpu_trace.collect().map_err(why)?;
                    let records = if MODES[mode].cpu() {
                        require(
                            gpu_records.is_empty(),
                            "CPU pipeline contains an unaccounted GPU native graph",
                        )?;
                        cpu_records
                    } else {
                        require(
                            cpu_records.is_empty(),
                            "GPU pipeline contains unaccounted CPU native work",
                        )?;
                        gpu_records
                    };
                    sample["trace"] = trace_record(records, MODES[mode])?;
                    sample["first_expert"] = json!(case.first);
                    samples.push(sample);
                }
                rounds.push(json!({"pair": pair, "pass": pass, "mode": MODES[mode].name(), "samples": samples,
                "conditioning": {"median_seconds": conditioning.median, "repetitions": conditioning.repetitions,
                    "rotation_bytes": conditioning.rotation_bytes}}));
            }
        }
    }
    let raw_bytes: usize = weights.iter().map(|w| w.raw.len()).sum();
    let native_bytes = |cuda: bool| -> Result<u64, String> {
        weights
            .iter()
            .map(|w| {
                (if cuda { w.gpu } else { w.cpu })
                    .canonical_byte_len(&w.shape)
                    .map_err(why)
            })
            .sum()
    };
    let result = json!({"status": if options.qualify_only { "qualification-complete" } else { "diagnostic-complete" },
        "canonical_only": true, "serving_speed_claim": false,
        "numerical_admission_claim": false, "features": provenance,
        "weights": weights.each_ref().map(|w| &w.identity), "misses": options.misses,
        "geometry": {"H": weights[0].shape[2], "F": weights[0].shape[1], "E": options.experts,
            "up_M": 1, "up_K": options.misses, "down_M": options.misses, "down_K": 1,
            "activation": "bf16", "result": "f32", "router_coefficients": "unit per-expert contributions",
            "cpu_result_transport": "canonical host readback/staging allocation and synchronous H2D",
            "gpu_directory": "mapped host control/placement/heartbeat/stamps/map/commits; device step and routes"},
        "footprint": {"raw_rotation_bytes": raw_bytes, "cpu_native_rotation_bytes": native_bytes(false)?,
            "gpu_native_rotation_bytes": native_bytes(true)?, "pinned_source_bytes": raw_bytes,
            "device_allocation_limit_bytes": 1_u64 << 30,
            "cpu_memory_usage": format!("{:?}", cpu.device.memory_usage()),
            "gpu_memory_usage": format!("{:?}", gpu.device.memory_usage())},
        "backends": {"cpu": backend_identity(&cpu), "gpu": backend_identity(&gpu)},
        "qualification": qualification, "cpu_cache_qualification": cache_qualification,
        "cpu_exchange_qualification": exchange_qualification, "rounds": rounds,
        "boundary": "Complete expert contribution from RAM: resident graphs, or CPU imports / GPU blocking uploads and bank conversion; CPU includes activation and result transport. File service, routing, serving and contention excluded.",
        "limitations": ["Readout vectors are not actual routed-layer activation captures", "Canonical implementations only; no tuning or exclusion of other qualified CPU configurations", "GPU is otherwise idle during CPU work; no overlap or model contention is represented", "Traced operator intervals are diagnostic and do not establish a generation speedup"]});
    std::fs::write(
        inputs.run.path("expert-crossover.json"),
        serde_json::to_vec_pretty(&result).map_err(why)?,
    )
    .map_err(why)?;
    Ok((
        result,
        vec![
            cpu.device.info().name.clone(),
            gpu.device.info().name.clone(),
        ],
    ))
}
