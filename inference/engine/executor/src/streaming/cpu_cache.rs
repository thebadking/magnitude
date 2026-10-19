//! Optional CPU-ready expert retention. This owns storage, not dispatch policy.
//!
//! A sealed source inventory fixes each layer's representation and quota. Fills
//! reserve capacity before allocation and publish complete immutable triples.
//! Weak storage observations keep retired allocations charged even when callers
//! retain tensor aliases after dropping a lease. No serving path enables it yet.

use crate::memory::{ClaimId, HoldingClass, MemoryBand, MemoryHeap, MemoryNeed, MemoryObservation};
use crate::platform::{self, MemoryReserves};
use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor, TensorStorageObserver};
use std::mem::size_of;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

static IDENTITIES: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuExpertLayerSpec {
    /// Complete immutable source identity, including model, tensor ranges and encodings.
    pub source_identity: String,
    pub experts: usize,
    /// Already admitted CPU representations; registration does not grant admission.
    pub elements: [Element; 3],
    /// Gate/up/down matrices, each [rows, columns], excluding the expert axis.
    pub matrices: [[u64; 2]; 3],
}

impl CpuExpertLayerSpec {
    pub fn record_bytes(&self) -> Result<u64, String> {
        if self.source_identity.is_empty()
            || self.experts == 0
            || self.matrices[0].contains(&0)
            || self.matrices[0] != self.matrices[1]
            || self.matrices[2] != [self.matrices[0][1], self.matrices[0][0]]
        {
            return Err("invalid CPU expert source geometry".into());
        }
        self.elements
            .iter()
            .zip(self.matrices)
            .try_fold(0_u64, |total, (element, [n, k])| {
                let bytes = element
                    .canonical_byte_len(&[1, n, k])
                    .map_err(|e| e.to_string())?;
                total
                    .checked_add(bytes)
                    .ok_or_else(|| "CPU expert record overflows".into())
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuExpertLayer {
    cache: u64,
    index: usize,
}

struct Record {
    weights: [Tensor; 3],
}

/// Keep this lease through physical completion of every consumer. Tensor aliases
/// are still accounted after lease release, but must never mutate cached weights.
#[derive(Clone)]
pub struct CpuExpertLease {
    _owner: Arc<CpuExpertCache>,
    record: Arc<Record>,
}
impl CpuExpertLease {
    pub fn weights(&self) -> &[Tensor; 3] {
        &self.record.weights
    }
}

struct Entry {
    record: Arc<Record>,
    stamp: u64,
}
struct Charge {
    observers: Vec<TensorStorageObserver>,
}
struct Pending {
    claim: Option<ClaimId>,
}
impl Charge {
    fn bytes(&self) -> u64 {
        self.observers
            .iter()
            .filter_map(TensorStorageObserver::charged_bytes)
            .sum()
    }
}
#[derive(Default)]
struct Slot {
    entry: Option<Entry>,
    charge: Option<Charge>,
    pending: Option<Pending>,
}
struct Layer {
    spec: CpuExpertLayerSpec,
    bytes: u64,
    quota: u64,
    slots: Vec<Slot>,
}
struct State {
    layers: Vec<Layer>,
    sealed: bool,
    clock: u64,
    generation: u64,
    heap: MemoryHeap,
}

impl State {
    fn prune(&mut self) {
        for slot in self.layers.iter_mut().flat_map(|layer| &mut layer.slots) {
            if slot
                .charge
                .as_ref()
                .is_some_and(|charge| charge.bytes() == 0)
            {
                debug_assert!(slot.entry.is_none());
                slot.charge = None;
            }
        }
    }
    fn live(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|layer| &layer.slots)
            .filter_map(|slot| slot.charge.as_ref())
            .map(Charge::bytes)
            .sum()
    }
    fn pending(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| {
                layer.bytes
                    * layer
                        .slots
                        .iter()
                        .filter(|slot| {
                            slot.pending
                                .as_ref()
                                .is_some_and(|pending| pending.claim.is_some())
                        })
                        .count() as u64
            })
            .sum()
    }
    fn controls(&self) -> u64 {
        let layers = self.layers.capacity() * size_of::<Layer>();
        let slots = self
            .layers
            .iter()
            .map(|layer| {
                layer.slots.capacity() * size_of::<Slot>() + layer.spec.source_identity.capacity()
            })
            .sum::<usize>();
        let records = self
            .layers
            .iter()
            .flat_map(|layer| &layer.slots)
            .map(|slot| {
                slot.charge.as_ref().map_or(0, |charge| {
                    charge.observers.capacity() * size_of::<TensorStorageObserver>()
                        + size_of::<Record>()
                        + 2 * size_of::<usize>()
                })
            })
            .sum::<usize>();
        (size_of::<Self>() + layers + slots + records) as u64
    }
    fn clear(&mut self) {
        self.generation = self
            .generation
            .checked_add(1)
            .expect("CPU cache generation exhausted");
        for slot in self.layers.iter_mut().flat_map(|layer| &mut layer.slots) {
            slot.entry = None;
        }
        self.prune();
    }
    fn seal(&mut self, limit: u64) -> Result<(), String> {
        if self.sealed {
            return Ok(());
        }
        let total = self.layers.iter().try_fold(0_u128, |total, layer| {
            total
                .checked_add(u128::from(layer.bytes) * layer.spec.experts as u128)
                .ok_or("CPU expert inventory overflows")
        })?;
        if total == 0 {
            return Err("empty CPU expert inventory".into());
        }
        for layer in &mut self.layers {
            let full = layer
                .bytes
                .checked_mul(layer.spec.experts as u64)
                .ok_or("CPU layer inventory overflows")?;
            layer.quota = (u128::from(limit) * u128::from(full) / total) as u64;
        }
        self.sealed = true;
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct CpuExpertCacheSnapshot {
    pub native_limit_bytes: u64,
    pub live_native_bytes: u64,
    pub reserved_native_bytes: u64,
    /// Cache-owned control structures; excludes Seismic's own tensor/program metadata.
    pub control_bytes: u64,
    pub retained_records: usize,
    pub retired_records: usize,
    pub pending_records: usize,
}

/// One host-pool cache shared by all physical owners of one target. The byte
/// limit covers native payloads; cache controls also receive surplus claims and
/// are reported separately. Caller-owned import staging/programs are not retained.
pub struct CpuExpertCache {
    identity: u64,
    catalog: DeviceCatalog,
    device: Device,
    reserves: MemoryReserves,
    protected_host_bytes: u64,
    native_limit: u64,
    state: Mutex<State>,
}

impl CpuExpertCache {
    pub fn new(
        catalog: DeviceCatalog,
        device: Device,
        reserves: MemoryReserves,
        protected_host_bytes: u64,
        native_limit: u64,
    ) -> Result<Arc<Self>, String> {
        if device.backend() != BackendName::Cpu || native_limit == 0 {
            return Err(
                "CPU expert cache requires a CPU device and a positive payload limit".into(),
            );
        }
        let identity = IDENTITIES
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| "CPU cache identity exhausted")?;
        Ok(Arc::new(Self {
            identity,
            catalog,
            device,
            reserves,
            protected_host_bytes,
            native_limit,
            state: Mutex::new(State {
                layers: Vec::new(),
                sealed: false,
                clock: 0,
                generation: 0,
                heap: MemoryHeap::new(),
            }),
        }))
    }

    fn claim(&self, state: &mut State, bytes: u64) -> Option<ClaimId> {
        let reading = platform::observe_host(&self.catalog, &self.reserves).ok()?;
        let available = reading
            .ceiling_bytes
            .saturating_sub(self.protected_host_bytes);
        let band = if available == 0 {
            MemoryBand::Reclaim
        } else {
            reading.band.into()
        };
        state
            .heap
            .observe(MemoryObservation {
                capacity_bytes: self.catalog.topology().host_pool().capacity_bytes,
                available_bytes: available,
                charged_bytes: state.live().saturating_add(state.controls()),
                band,
            })
            .ok()?;
        state
            .heap
            .claim(MemoryNeed {
                minimum_bytes: bytes,
                preferred_bytes: bytes,
                class: HoldingClass::Surplus,
            })
            .ok()
            .map(|claim| claim.id)
    }

    pub fn register_layer(&self, spec: CpuExpertLayerSpec) -> Result<CpuExpertLayer, String> {
        let bytes = spec.record_bytes()?;
        bytes
            .checked_mul(spec.experts as u64)
            .ok_or("CPU layer inventory overflows")?;
        let mut state = self.state.lock().expect("CPU expert cache");
        if let Some(index) = state
            .layers
            .iter()
            .position(|layer| layer.spec.source_identity == spec.source_identity)
        {
            if state.layers[index].spec != spec {
                return Err("CPU source identity changed geometry".into());
            }
            return Ok(CpuExpertLayer {
                cache: self.identity,
                index,
            });
        }
        if state.sealed {
            return Err("CPU expert inventory changed after use".into());
        }
        let metadata = spec
            .experts
            .checked_mul(size_of::<Slot>())
            .and_then(|bytes| bytes.checked_add(size_of::<Layer>()))
            .ok_or("CPU cache metadata overflows")? as u64;
        let claim = self
            .claim(&mut state, metadata)
            .ok_or("host claim refused CPU cache inventory")?;
        let mut slots = Vec::new();
        let allocated = slots
            .try_reserve_exact(spec.experts)
            .and_then(|_| state.layers.try_reserve_exact(1));
        state
            .heap
            .cancel_claim(claim)
            .map_err(|e| format!("CPU inventory claim: {e:?}"))?;
        allocated.map_err(|e| e.to_string())?;
        slots.resize_with(spec.experts, Slot::default);
        let index = state.layers.len();
        state.layers.push(Layer {
            spec,
            bytes,
            quota: 0,
            slots,
        });
        Ok(CpuExpertLayer {
            cache: self.identity,
            index,
        })
    }

    fn validate(&self, state: &State, layer: CpuExpertLayer, expert: usize) -> Result<(), String> {
        if layer.cache != self.identity
            || state
                .layers
                .get(layer.index)
                .is_none_or(|layer| expert >= layer.slots.len())
        {
            return Err("CPU expert key belongs to another cache or is out of bounds".into());
        }
        Ok(())
    }

    fn hit(
        owner: &Arc<Self>,
        state: &mut State,
        layer: usize,
        expert: usize,
    ) -> Result<Option<CpuExpertLease>, String> {
        state.clock = state
            .clock
            .checked_add(1)
            .ok_or("CPU cache clock exhausted")?;
        let clock = state.clock;
        Ok(state.layers[layer].slots[expert]
            .entry
            .as_mut()
            .map(|entry| {
                entry.stamp = clock;
                CpuExpertLease {
                    _owner: owner.clone(),
                    record: entry.record.clone(),
                }
            }))
    }

    pub fn get(
        self: &Arc<Self>,
        layer: CpuExpertLayer,
        expert: usize,
    ) -> Result<Option<CpuExpertLease>, String> {
        let mut state = self.state.lock().expect("CPU expert cache");
        self.validate(&state, layer, expert)?;
        state.seal(self.native_limit)?;
        Self::hit(self, &mut state, layer.index, expert)
    }

    /// Reserve before allocating, fill outside the mutex, then publish the
    /// triple atomically. A refused optional fill calls no builder. The builder
    /// writes the supplied tensors completely and never replaces their storage.
    pub fn get_or_fill(
        self: &Arc<Self>,
        layer: CpuExpertLayer,
        expert: usize,
        fill: impl FnOnce(&mut [Tensor; 3]) -> Result<(), String>,
    ) -> Result<Option<CpuExpertLease>, String> {
        let (spec, generation) = {
            let mut state = self.state.lock().expect("CPU expert cache");
            self.validate(&state, layer, expert)?;
            state.seal(self.native_limit)?;
            if let Some(lease) = Self::hit(self, &mut state, layer.index, expert)? {
                return Ok(Some(lease));
            }
            state.prune();
            let slot = &state.layers[layer.index].slots[expert];
            if slot.pending.is_some() || slot.charge.is_some() {
                return Ok(None);
            }
            let required = state.layers[layer.index].bytes;
            if required > state.layers[layer.index].quota {
                return Ok(None);
            }
            loop {
                let owner = &state.layers[layer.index];
                let held = owner
                    .slots
                    .iter()
                    .map(|slot| {
                        slot.charge.as_ref().map_or(0, Charge::bytes)
                            + if slot
                                .pending
                                .as_ref()
                                .is_some_and(|pending| pending.claim.is_some())
                            {
                                owner.bytes
                            } else {
                                0
                            }
                    })
                    .sum::<u64>();
                if held
                    .checked_add(required)
                    .is_some_and(|sum| sum <= owner.quota)
                {
                    break;
                }
                let victim = owner
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(index, slot)| {
                        slot.entry
                            .as_ref()
                            .filter(|entry| Arc::strong_count(&entry.record) == 1)
                            .map(|entry| (entry.stamp, index))
                    })
                    .min()
                    .map(|(_, index)| index);
                let Some(victim) = victim else {
                    return Ok(None);
                };
                state.layers[layer.index].slots[victim].entry = None;
                state.prune();
            }
            let overhead = (size_of::<Record>()
                + 2 * size_of::<usize>()
                + 3 * size_of::<TensorStorageObserver>()) as u64;
            let Some(claim) = self.claim(
                &mut state,
                required
                    .checked_add(overhead)
                    .ok_or("CPU fill claim overflow")?,
            ) else {
                state.clear();
                return Ok(None);
            };
            state.layers[layer.index].slots[expert].pending = Some(Pending { claim: Some(claim) });
            (state.layers[layer.index].spec.clone(), state.generation)
        };
        let mut guard = FillGuard {
            cache: self.clone(),
            layer: layer.index,
            expert,
            observers: Vec::with_capacity(3),
            active: true,
        };
        let mut record = {
            // Allocation and conversion are separate phases. Reconcile the
            // reservation before another claim observes already-backed bytes,
            // so fresh physical headroom never pays the reservation twice.
            let mut state = self.state.lock().expect("CPU expert cache");
            if state.generation != generation {
                return Ok(None);
            }
            let mut outputs = Vec::with_capacity(3);
            for kind in 0..3 {
                let [n, k] = spec.matrices[kind];
                let tensor = Tensor::zeros(&self.device, spec.elements[kind], &[1, n, k])
                    .map_err(|e| e.to_string())?;
                guard.observers.push(tensor.observe_storage());
                outputs.push(tensor);
            }
            let outputs: [Tensor; 3] = outputs.try_into().map_err(|_| "CPU triple count")?;
            let actual = guard
                .observers
                .iter()
                .filter_map(TensorStorageObserver::charged_bytes)
                .sum::<u64>();
            if actual != spec.record_bytes()? {
                return Err("CPU allocation differs from its canonical reservation".into());
            }
            let record = Arc::new(Record { weights: outputs });
            let slot = &mut state.layers[layer.index].slots[expert];
            slot.charge = Some(Charge {
                observers: record.weights.iter().map(Tensor::observe_storage).collect(),
            });
            let claim = slot
                .pending
                .as_mut()
                .expect("CPU fill reservation")
                .claim
                .take()
                .expect("unbacked CPU claim");
            state
                .heap
                .cancel_claim(claim)
                .map_err(|e| format!("CPU backed claim: {e:?}"))?;
            record
        };
        fill(&mut Arc::get_mut(&mut record).expect("private CPU fill").weights)?;
        for kind in 0..3 {
            let [n, k] = spec.matrices[kind];
            let output = &record.weights[kind];
            if output.element() != spec.elements[kind]
                || output.extents() != [1, n, k]
                || !output.belongs_to(&self.device)
                || output.observe_storage().identity() != guard.observers[kind].identity()
            {
                return Err("CPU fill replaced its admitted storage or descriptor".into());
            }
        }
        let mut state = self.state.lock().expect("CPU expert cache");
        if state.generation != generation {
            return Ok(None);
        }
        state.clock = state
            .clock
            .checked_add(1)
            .ok_or("CPU cache clock exhausted")?;
        let stamp = state.clock;
        let slot = &mut state.layers[layer.index].slots[expert];
        let pending = slot.pending.take().expect("CPU fill reservation");
        debug_assert!(pending.claim.is_none());
        guard.active = false;
        state.layers[layer.index].slots[expert].entry = Some(Entry {
            record: record.clone(),
            stamp,
        });
        Ok(Some(CpuExpertLease {
            _owner: self.clone(),
            record,
        }))
    }

    pub fn clear(&self) {
        self.state.lock().expect("CPU expert cache").clear();
    }

    pub fn reclaim_if_needed(&self) {
        let normal = platform::observe_host(&self.catalog, &self.reserves).is_ok_and(|reading| {
            reading.band == platform::MemoryBand::Normal
                && reading.ceiling_bytes > self.protected_host_bytes
        });
        if !normal {
            self.clear();
        }
    }

    pub fn snapshot(&self) -> CpuExpertCacheSnapshot {
        let mut state = self.state.lock().expect("CPU expert cache");
        state.prune();
        let (retained, retired, pending) = state.layers.iter().flat_map(|layer| &layer.slots).fold(
            (0, 0, 0),
            |(retained, retired, pending), slot| {
                (
                    retained + usize::from(slot.entry.is_some()),
                    retired + usize::from(slot.entry.is_none() && slot.charge.is_some()),
                    pending + usize::from(slot.pending.is_some()),
                )
            },
        );
        CpuExpertCacheSnapshot {
            native_limit_bytes: self.native_limit,
            live_native_bytes: state.live(),
            reserved_native_bytes: state.pending(),
            control_bytes: state.controls(),
            retained_records: retained,
            retired_records: retired,
            pending_records: pending,
        }
    }
}

struct FillGuard {
    cache: Arc<CpuExpertCache>,
    layer: usize,
    expert: usize,
    observers: Vec<TensorStorageObserver>,
    active: bool,
}
impl Drop for FillGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut state = self.cache.state.lock().expect("CPU expert cache");
        let slot = &mut state.layers[self.layer].slots[self.expert];
        let pending = slot.pending.take().expect("CPU fill reservation");
        if slot.charge.is_none() {
            slot.charge = Some(Charge {
                observers: std::mem::take(&mut self.observers),
            });
        }
        if let Some(claim) = pending.claim {
            let _ = state.heap.cancel_claim(claim);
        }
        state.prune();
    }
}
