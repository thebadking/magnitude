//! One CPU execution lane and one bounded prepared-expert cache per target.
//! Both device watchers share this owner; layer registration precedes readiness.
use super::cpu_cache::{CpuExpertCache, CpuExpertLayer, CpuExpertLayerSpec, CpuExpertLease};
use super::cpu_programs::{CpuExpertEntry, CpuExpertPrograms};
use crate::memory::{HoldingClass, MemoryHeap, MemoryNeed, MemoryObservation};
use crate::planning::CpuExpertBinding;
use crate::platform::{self, MemoryReserves};
use magnitude_kernels::{repack_weight, routed_cpu_expert};
use seismic::{DeviceCatalog, Element, Tensor};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone)]
pub(crate) struct CpuExpertLayerContext {
    key: CpuExpertLayer,
    program: Arc<CpuExpertEntry>,
}

struct Workspace {
    raw: [Tensor; 3],
    input: Tensor,
    output: Tensor,
    heap: MemoryHeap,
}

pub struct CpuExpertService {
    programs: Arc<CpuExpertPrograms>,
    cache: Arc<CpuExpertCache>,
    catalog: DeviceCatalog,
    reserves: MemoryReserves,
    protected_host_bytes: u64,
    standing_bytes: u64,
    scratch_bytes: u64,
    lane: Mutex<Workspace>,
    pub(crate) eligible: AtomicU64,
    pub(crate) hits: AtomicU64,
    fills: AtomicU64,
}

impl CpuExpertService {
    pub fn new(
        catalog: DeviceCatalog,
        programs: Arc<CpuExpertPrograms>,
        reserves: MemoryReserves,
        protected_host_bytes: u64,
    ) -> Result<Arc<Self>, String> {
        let mut raw_sizes = [0_u64; 3];
        let mut width = 0;
        let mut scratch = 0;
        for binding in &programs.plan.bindings {
            width = width.max(binding.hidden());
            scratch = scratch.max(binding.invocation_scratch_bytes()?);
            for kind in 0..3 {
                raw_sizes[kind] = raw_sizes[kind].max(binding.source_bytes(kind)?);
            }
        }
        let persistent = raw_sizes
            .iter()
            .try_fold(0_u64, |sum, b| {
                sum.checked_add(*b).ok_or("CPU staging charge overflow")
            })?
            .checked_add(
                width
                    .checked_mul(6)
                    .ok_or("CPU input/output charge overflow")?,
            )
            .ok_or("CPU workspace charge overflow")?;
        let total = persistent
            .checked_add(scratch)
            .ok_or("CPU workspace charge overflow")?;
        if total != programs.plan.workspace_bytes()? {
            return Err("CPU workspace differs from its checked plan".into());
        }
        let mut heap = MemoryHeap::new();
        let reading = platform::observe_host(&catalog, &reserves).map_err(|e| e.to_string())?;
        heap.observe(MemoryObservation {
            capacity_bytes: catalog.topology().host_pool().capacity_bytes,
            available_bytes: reading.ceiling_bytes.saturating_sub(protected_host_bytes),
            charged_bytes: programs.plan.program_bytes()?,
            band: reading.band.into(),
        })
        .map_err(|e| format!("CPU workspace observation: {e:?}"))?;
        let claim = heap
            .claim(MemoryNeed {
                minimum_bytes: total,
                preferred_bytes: total,
                class: HoldingClass::Model,
            })
            .map_err(|e| format!("CPU workspace admission: {e:?}"))?;
        let allocate = |bytes| {
            Tensor::zeros(&programs.device, Element::bool(), &[bytes]).map_err(|e| e.to_string())
        };
        let raw = [
            allocate(raw_sizes[0])?,
            allocate(raw_sizes[1])?,
            allocate(raw_sizes[2])?,
        ];
        let input = allocate(width * 2)?;
        let output = allocate(width * 4)?;
        heap.cancel_claim(claim.id)
            .map_err(|e| format!("CPU workspace reconciliation: {e:?}"))?;
        let standing_bytes = persistent
            .checked_add(programs.plan.program_bytes()?)
            .ok_or("CPU standing charge overflow")?;
        let cache = CpuExpertCache::new(
            catalog.clone(),
            programs.device.clone(),
            reserves,
            protected_host_bytes
                .checked_add(scratch)
                .ok_or("CPU protected charge overflow")?,
            programs.plan.native_limit_bytes,
        )?;
        Ok(Arc::new(Self {
            programs,
            cache,
            catalog,
            reserves,
            protected_host_bytes,
            standing_bytes,
            scratch_bytes: scratch,
            lane: Mutex::new(Workspace {
                raw,
                input,
                output,
                heap,
            }),
            eligible: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            fills: AtomicU64::new(0),
        }))
    }

    pub(crate) fn register(
        &self,
        binding: &CpuExpertBinding,
        source_identity: String,
        experts: usize,
    ) -> Result<CpuExpertLayerContext, String> {
        let program = self
            .programs
            .entries
            .iter()
            .find(|entry| entry.binding == *binding)
            .cloned()
            .ok_or("CPU layer has no prepared checked program")?;
        let key = self.cache.register_layer(CpuExpertLayerSpec {
            source_identity,
            experts,
            elements: binding.resident,
            matrices: binding.matrices,
        })?;
        Ok(CpuExpertLayerContext { key, program })
    }
    pub(crate) fn get(
        &self,
        layer: &CpuExpertLayerContext,
        expert: usize,
    ) -> Result<Option<CpuExpertLease>, String> {
        self.cache.get(layer.key, expert)
    }
    fn admitted<T>(
        &self,
        workspace: &mut Workspace,
        run: impl FnOnce(&mut Workspace) -> Result<T, String>,
    ) -> Result<T, String> {
        let reading =
            platform::observe_host(&self.catalog, &self.reserves).map_err(|e| e.to_string())?;
        workspace
            .heap
            .observe(MemoryObservation {
                capacity_bytes: self.catalog.topology().host_pool().capacity_bytes,
                available_bytes: reading
                    .ceiling_bytes
                    .saturating_sub(self.protected_host_bytes),
                charged_bytes: self.standing_bytes,
                band: reading.band.into(),
            })
            .map_err(|e| format!("CPU invocation observation: {e:?}"))?;
        let claim = workspace
            .heap
            .claim(MemoryNeed {
                minimum_bytes: self.scratch_bytes,
                preferred_bytes: self.scratch_bytes,
                class: HoldingClass::InFlight,
            })
            .map_err(|e| format!("CPU invocation admission: {e:?}"))?;
        let result = run(workspace);
        workspace
            .heap
            .cancel_claim(claim.id)
            .map_err(|e| format!("CPU invocation reconciliation: {e:?}"))?;
        result
    }
    pub(crate) fn execute(
        &self,
        layer: &CpuExpertLayerContext,
        lease: &CpuExpertLease,
        input: &[u8],
    ) -> Result<Vec<u8>, String> {
        let mut workspace = self
            .lane
            .try_lock()
            .map_err(|_| "CPU expert lane unavailable")?;
        self.admitted(&mut workspace, |workspace| {
            let h = layer.program.binding.hidden();
            let mut normalized = workspace
                .input
                .region(Element::bf16(), &[1, h], 0)
                .map_err(|e| e.to_string())?;
            normalized
                .write_from_host(input)
                .map_err(|e| e.to_string())?;
            let mut output = workspace
                .output
                .region(Element::f32(), &[1, h], 0)
                .map_err(|e| e.to_string())?;
            let weights = lease.weights();
            layer
                .program
                .contribution
                .call_into(
                    routed_cpu_expert::Args {
                        normalized: &normalized,
                        expert_gate: &weights[0],
                        expert_up: &weights[1],
                        expert_down: &weights[2],
                    },
                    routed_cpu_expert::OutputArgs { value: &mut output },
                )
                .map_err(|e| e.to_string())?;
            output.read_to_host().map_err(|e| e.to_string())
        })
    }
    pub(crate) fn retain(
        &self,
        layer: &CpuExpertLayerContext,
        expert: usize,
        raw: [&[u8]; 3],
    ) -> Result<bool, String> {
        if self.cache.get(layer.key, expert)?.is_some() {
            return Ok(false);
        }
        let mut workspace = match self.lane.try_lock() {
            Ok(workspace) => workspace,
            Err(_) => return Ok(false),
        };
        let entry = &layer.program;
        let held = self.cache.get_or_fill(layer.key, expert, |outputs| {
            self.admitted(&mut workspace, |workspace| {
                for kind in 0..3 {
                    let matrix = entry.binding.matrices[kind];
                    let mut source = workspace.raw[kind]
                        .region(entry.binding.source[kind], &[1, matrix[0], matrix[1]], 0)
                        .map_err(|e| e.to_string())?;
                    source
                        .write_from_host(raw[kind])
                        .map_err(|e| e.to_string())?;
                    entry.import[kind]
                        .call_into(
                            repack_weight::Args { source: &source },
                            repack_weight::OutputArgs {
                                value: &mut outputs[kind],
                            },
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            })
        })?;
        if held.is_some() {
            self.fills.fetch_add(1, Ordering::Relaxed);
        }
        Ok(held.is_some())
    }
    pub fn allocated_bytes(&self) -> u64 {
        let snapshot = self.cache.snapshot();
        self.standing_bytes
            .saturating_add(snapshot.live_native_bytes)
            .saturating_add(snapshot.control_bytes)
    }
    pub fn reclaim_if_needed(&self) {
        self.cache.reclaim_if_needed();
    }
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({"cache":self.cache.snapshot(),"standing_bytes":self.standing_bytes,"invocation_scratch_bound":self.scratch_bytes,
            "eligible":self.eligible.load(Ordering::Relaxed),"hits":self.hits.load(Ordering::Relaxed),"fills":self.fills.load(Ordering::Relaxed)})
    }
}
