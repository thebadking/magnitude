//! Header-only optional CPU expert inventory. Ranges retain the complete
//! program inventory; each physical bank still owns only its local channels.
use super::{ExpertBankPlan, weights::BankLayer};
use seismic::Element;

pub fn requested_cpu_expert_bytes() -> Result<u64, String> {
    match std::env::var("MAGNITUDE_CPU_EXPERT_CACHE_BYTES") {
        Err(std::env::VarError::NotPresent) => Ok(0),
        Ok(value) => value
            .parse()
            .map_err(|_| "MAGNITUDE_CPU_EXPERT_CACHE_BYTES must be an unsigned byte count".into()),
        Err(_) => Err("invalid MAGNITUDE_CPU_EXPERT_CACHE_BYTES".into()),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuExpertBinding {
    pub source: [Element; 3],
    pub resident: [Element; 3],
    pub matrices: [[u64; 2]; 3],
}

impl CpuExpertBinding {
    pub fn of(bank: &ExpertBankPlan, layer: &BankLayer) -> Option<Self> {
        let source = layer.tensors.map(|tensor| bank.pools[tensor.pool].source);
        let mut resident = [Element::f32(); 3];
        for kind in 0..3 {
            let name = match source[kind].name() {
                "gguf_q4_k" => "q4k@rows8",
                "gguf_q5_k" => "q5k@rows8",
                "gguf_q5_0" => "q5g32s@rows8",
                "gguf_q5_1" => "q5g32@rows8",
                "gguf_q8_0" => "q8g32s@rows8",
                _ => return None,
            };
            resident[kind] = Element::named(name)?;
        }
        let matrices = layer.tensors.map(|tensor| bank.pools[tensor.pool].matrix);
        let [f, h] = matrices[0];
        if h == 0 || f == 0 || matrices[1] != [f, h] || matrices[2] != [h, f] {
            return None;
        }
        Some(Self {
            source,
            resident,
            matrices,
        })
    }
    pub fn hidden(&self) -> u64 {
        self.matrices[0][1]
    }
    pub fn features(&self) -> u64 {
        self.matrices[0][0]
    }
    pub fn source_bytes(&self, kind: usize) -> Result<u64, String> {
        self.source[kind]
            .canonical_byte_len(&[1, self.matrices[kind][0], self.matrices[kind][1]])
            .map_err(|e| e.to_string())
    }
    pub fn exchange_bytes(&self) -> Result<u64, String> {
        self.hidden()
            .checked_mul(6)
            .and_then(|b| b.checked_add(20))
            .ok_or("CPU exchange charge overflow".into())
    }
    pub fn invocation_scratch_bytes(&self) -> Result<u64, String> {
        use magnitude_kernels::routed_cpu_expert;
        let mut graph = seismic::NativeGraphMetadata::new(seismic::BackendName::Cpu);
        let x = graph
            .port(Element::bf16(), &[1, self.hidden()])
            .map_err(|e| e.to_string())?;
        let gate = graph
            .port(self.resident[0], &[1, self.features(), self.hidden()])
            .map_err(|e| e.to_string())?;
        let up = graph
            .port(self.resident[1], &[1, self.features(), self.hidden()])
            .map_err(|e| e.to_string())?;
        let down = graph
            .port(self.resident[2], &[1, self.hidden(), self.features()])
            .map_err(|e| e.to_string())?;
        let result = graph
            .enqueue::<routed_cpu_expert::Entry>(
                &[
                    ("EGW", self.resident[0]),
                    ("EUW", self.resident[1]),
                    ("EDW", self.resident[2]),
                ],
                &[("H", self.hidden()), ("F", self.features())],
                routed_cpu_expert::WorkflowArgs {
                    normalized: x.tensor().into(),
                    expert_gate: gate.tensor().into(),
                    expert_up: up.tensor().into(),
                    expert_down: down.tensor().into(),
                },
            )
            .map_err(|e| e.to_string())?
            .value;
        graph.export(&result).map_err(|e| e.to_string())?;
        Ok(graph.seal().map_err(|e| e.to_string())?.workspace)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuExpertPlan {
    pub native_limit_bytes: u64,
    pub bindings: Vec<CpuExpertBinding>,
}

impl CpuExpertPlan {
    pub(crate) fn of(bank: &ExpertBankPlan) -> Result<Option<Self>, String> {
        let limit = requested_cpu_expert_bytes()?;
        if limit == 0 {
            return Ok(None);
        }
        if bank.selected < 2
            || bank
                .layers
                .iter()
                .flat_map(|layer| &layer.resident)
                .any(|weight| weight.resident.layout() != seismic::Layout::Mma16)
        {
            return Err(
                "CPU expert dispatch requires a CUDA streamed bank selecting at least two experts"
                    .into(),
            );
        }
        let mut bindings = Vec::new();
        for layer in &bank.layers {
            if let Some(binding) = CpuExpertBinding::of(bank, layer) {
                if !bindings.contains(&binding) {
                    bindings.push(binding);
                }
            }
        }
        if bindings.is_empty() {
            return Err("CPU expert dispatch has no supported source bindings".into());
        }
        Ok(Some(Self {
            native_limit_bytes: limit,
            bindings,
        }))
    }

    /// One serial CPU lane, shared by both GPU watchers. Source arenas and
    /// input/output tensors are persistent; the invocation scratch is bounded
    /// by its checked entry and kept as a separate transient allowance.
    pub fn workspace_bytes(&self) -> Result<u64, String> {
        let mut source = [0_u64; 3];
        let mut hidden = 0;
        let mut scratch = 0;
        for binding in &self.bindings {
            for kind in 0..3 {
                source[kind] = source[kind].max(binding.source_bytes(kind)?);
            }
            hidden = hidden.max(binding.hidden());
            scratch = scratch.max(binding.invocation_scratch_bytes()?);
        }
        source
            .into_iter()
            .try_fold(0_u64, |total, b| {
                total
                    .checked_add(b)
                    .ok_or("CPU source arena charge overflow")
            })?
            .checked_add(hidden.checked_mul(6).ok_or("CPU operand charge overflow")?)
            .and_then(|b| b.checked_add(scratch))
            .ok_or("CPU workspace charge overflow".into())
    }
    pub fn program_bytes(&self) -> Result<u64, String> {
        use magnitude_kernels::{repack_weight, routed_cpu_expert};
        let mut imports = std::collections::HashSet::new();
        for binding in &self.bindings {
            for kind in 0..3 {
                imports.insert((binding.source[kind], binding.resident[kind]));
            }
        }
        let contributions = (self.bindings.len() as u64).checked_mul(seismic::NativeKernel::<routed_cpu_expert::Entry>::planned_invocation_workspace_bytes()).ok_or("CPU program charge overflow")?;
        (imports.len() as u64)
            .checked_mul(
                seismic::NativeKernel::<repack_weight::Entry>::planned_invocation_workspace_bytes(),
            )
            .and_then(|bytes| bytes.checked_add(contributions))
            .ok_or("CPU program charge overflow".into())
    }
}
