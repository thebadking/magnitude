//! A load's optional, shared host cache of immutable compressed expert records.
//! File identities and complete tensor record geometry are registered before
//! use. Layer quotas prevent a later decoder range from evicting the first;
//! bulk scans can read retained records without filling the decode cache.

use crate::memory::{HoldingClass, MemoryBand, MemoryHeap, MemoryNeed, MemoryObservation};
use crate::platform::{self, MemoryReserves};
use magnitude_family_contracts::WeightScope;
use seismic::DeviceCatalog;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Access { Demand, Bulk }

pub(super) struct Record {
    bytes: Vec<u8>,
    allocated: Arc<AtomicU64>,
    group: Arc<AtomicU64>,
    charge: u64,
}

impl Record {
    pub(super) fn bytes(&self) -> &[u8] { &self.bytes }
}

impl Drop for Record {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.bytes));
        self.allocated.fetch_sub(self.charge, Ordering::AcqRel);
        self.group.fetch_sub(self.charge, Ordering::AcqRel);
    }
}

#[derive(Default)]
struct Entry { record: Option<Arc<Record>>, stamp: u64, device_resident: bool }
struct TensorRecords { file: usize, base: u64, bytes: usize, group: usize, entries: Vec<Entry> }
struct Group { scope: WeightScope, source_bytes: u64, allocated: Arc<AtomicU64> }
struct State {
    files: Vec<PathBuf>, tensors: Vec<TensorRecords>, groups: Vec<Group>,
    clock: u64, limit: Option<u64>, heap: MemoryHeap,
}

/// One cache is shared by every physical owner of the same loaded target.
/// Payload allocation uses the host pool's normal claims, independently of
/// the GPU ledgers. The census attributes the cache once to its first owner.
pub struct HostExpertCache {
    catalog: DeviceCatalog,
    reserves: MemoryReserves,
    protected_host_bytes: u64,
    requested: Option<u64>,
    state: Mutex<State>,
    allocated: Arc<AtomicU64>,
    metadata: AtomicU64,
}

impl HostExpertCache {
    pub fn new(catalog: DeviceCatalog, reserves: MemoryReserves, protected_host_bytes: u64)
        -> Result<Option<Arc<Self>>, String>
    {
        // Opt in only: explicit retention competes with the platform's file
        // cache, so more retained RAM does not by itself imply faster serving.
        let requested = match std::env::var("MAGNITUDE_HOST_EXPERT_CACHE_BYTES") {
            Ok(value) if value == "auto" => None,
            Ok(value) => Some(value.parse::<u64>()
                .map_err(|_| "MAGNITUDE_HOST_EXPERT_CACHE_BYTES must be auto or an unsigned byte count".to_owned())?),
            Err(std::env::VarError::NotPresent) => Some(0),
            Err(std::env::VarError::NotUnicode(_)) => return Err("invalid MAGNITUDE_HOST_EXPERT_CACHE_BYTES".into()),
        };
        if requested == Some(0) { return Ok(None); }
        Ok(Some(Arc::new(Self { catalog, reserves, protected_host_bytes, requested,
            allocated: Arc::new(AtomicU64::new(0)),
            metadata: AtomicU64::new(0),
            state: Mutex::new(State { files: Vec::new(), tensors: Vec::new(), groups: Vec::new(),
                clock: 0, limit: None, heap: MemoryHeap::new() }) })))
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.allocated.load(Ordering::Acquire) + self.metadata.load(Ordering::Acquire)
    }

    pub fn reclaim_if_needed(&self) {
        let normal = platform::observe_host(&self.catalog, &self.reserves)
            .is_ok_and(|reading| reading.band == platform::MemoryBand::Normal
                && reading.ceiling_bytes > self.protected_host_bytes);
        if !normal { self.clear(); }
    }

    pub fn clear(&self) {
        let mut state = self.state.lock().expect("host expert cache");
        Self::clear_records(&mut state);
    }

    fn clear_records(state: &mut State) {
        for tensor in &mut state.tensors {
            for entry in &mut tensor.entries { entry.record = None; }
        }
    }

    pub(crate) fn register_file(&self, path: &Path) -> usize {
        let mut state = self.state.lock().expect("host expert cache");
        if let Some(index) = state.files.iter().position(|file| file == path) { return index; }
        let index = state.files.len(); state.files.push(path.to_owned()); index
    }

    pub(crate) fn register_tensor(&self, path: &Path, base: u64, bytes: u64, experts: u64, scope: WeightScope)
        -> Result<(), String>
    {
        let file = self.register_file(path);
        let mut state = self.state.lock().expect("host expert cache");
        let bytes = usize::try_from(bytes).map_err(|_| "host expert record exceeds usize")?;
        let experts = usize::try_from(experts).map_err(|_| "host expert count exceeds usize")?;
        if bytes == 0 || experts == 0 { return Err("empty host expert tensor".into()); }
        if let Some(old) = state.tensors.iter().find(|tensor| tensor.file == file && tensor.base == base) {
            return if old.bytes == bytes && old.entries.len() == experts { Ok(()) }
                else { Err("host expert tensor geometry changed".into()) };
        }
        if state.limit.is_some() { return Err("host expert inventory changed after reads began".into()); }
        let source_bytes = (bytes as u64).checked_mul(experts as u64).ok_or("host expert bytes overflow")?;
        base.checked_add(source_bytes).ok_or("host expert file range overflow")?;
        let group = match state.groups.iter().position(|group| group.scope == scope) {
            Some(index) => index,
            None => {
                let index = state.groups.len();
                state.groups.push(Group { scope, source_bytes: 0, allocated: Arc::new(AtomicU64::new(0)) }); index
            }
        };
        let metadata = (experts as u64).checked_mul(std::mem::size_of::<Entry>() as u64)
            .ok_or("host expert metadata overflow")?;
        let Some(claim) = self.claim(&mut state, metadata) else { return Ok(()); };
        let mut entries = Vec::new();
        let allocated = entries.try_reserve_exact(experts);
        state.heap.cancel_claim(claim).map_err(|error| format!("host cache metadata claim: {error:?}"))?;
        if allocated.is_err() { return Ok(()); }
        entries.resize_with(experts, Entry::default);
        state.groups[group].source_bytes = state.groups[group].source_bytes.checked_add(source_bytes)
            .ok_or("host expert group bytes overflow")?;
        state.tensors.push(TensorRecords { file, base, bytes, group, entries });
        let metadata = state.tensors.capacity() * std::mem::size_of::<TensorRecords>()
            + state.groups.capacity() * std::mem::size_of::<Group>()
            + state.files.capacity() * std::mem::size_of::<PathBuf>()
            + state.tensors.iter().map(|tensor| tensor.entries.capacity() * std::mem::size_of::<Entry>()).sum::<usize>()
            + state.files.iter().map(|file| file.as_os_str().as_encoded_bytes().len()).sum::<usize>();
        self.metadata.store(metadata as u64, Ordering::Release);
        Ok(())
    }

    fn locate(state: &State, file: usize, offset: u64, bytes: usize) -> Option<(usize, usize)> {
        state.tensors.iter().enumerate().find_map(|(index, tensor)| {
            if tensor.file != file || tensor.bytes != bytes { return None; }
            let relative = offset.checked_sub(tensor.base)?;
            if relative % bytes as u64 != 0 { return None; }
            let expert = usize::try_from(relative / bytes as u64).ok()?;
            (expert < tensor.entries.len()).then_some((index, expert))
        })
    }

    pub(super) fn get(&self, file: usize, offset: u64, bytes: usize, access: Access) -> Option<Arc<Record>> {
        let mut state = self.state.lock().expect("host expert cache");
        let (tensor, expert) = Self::locate(&state, file, offset, bytes)?;
        state.clock += 1;
        let stamp = state.clock;
        let entry = &mut state.tensors[tensor].entries[expert];
        let record = entry.record.clone()?;
        if access == Access::Demand { entry.stamp = stamp; }
        Some(record)
    }

    /// Residency is a replacement hint, never authority to read device bytes.
    /// Prefer keeping records whose next use would otherwise need the file.
    pub(super) fn device_residency(&self, file: usize, offset: u64, bytes: usize, resident: bool) {
        let mut state = self.state.lock().expect("host expert cache");
        if let Some((tensor, expert)) = Self::locate(&state, file, offset, bytes) {
            state.tensors[tensor].entries[expert].device_resident = resident;
        }
    }

    fn claim(&self, state: &mut State, bytes: u64) -> Option<crate::memory::ClaimId> {
        let reading = platform::observe_host(&self.catalog, &self.reserves).ok()?;
        let available = reading.ceiling_bytes.saturating_sub(self.protected_host_bytes);
        let band = if available == 0 { MemoryBand::Reclaim } else { reading.band.into() };
        state.heap.observe(MemoryObservation {
            capacity_bytes: self.catalog.topology().host_pool().capacity_bytes,
            available_bytes: available, charged_bytes: self.allocated_bytes(), band,
        }).ok()?;
        state.heap.claim(MemoryNeed { minimum_bytes: bytes, preferred_bytes: bytes,
            class: HoldingClass::Surplus }).ok().map(|claim| claim.id)
    }

    pub(super) fn remember(&self, file: usize, offset: u64, bytes: &[u8], access: Access) -> bool {
        // Prompt scans touch most experts once. Retaining that scan duplicates
        // the file cache and spends the decode budget before demand exists.
        if access == Access::Bulk { return false; }
        let mut state = self.state.lock().expect("host expert cache");
        let Some((tensor, expert)) = Self::locate(&state, file, offset, bytes.len()) else { return false; };
        if state.tensors[tensor].entries[expert].record.is_some() { return true; }
        let total = state.groups.iter().map(|group| group.source_bytes).sum::<u64>();
        if state.limit.is_none() {
            let Ok(reading) = platform::observe_host(&self.catalog, &self.reserves) else { return false; };
            if reading.band != platform::MemoryBand::Normal { return false; }
            let available = reading.ceiling_bytes.saturating_sub(self.protected_host_bytes);
            let limit = self.requested.unwrap_or(available / 4 * 3).min(total);
            state.limit = Some(limit);
            eprintln!("host expert cache: {} bytes, {} routed layer shares", limit, state.groups.len());
        }
        let group = state.tensors[tensor].group;
        let quota = (u128::from(state.limit.unwrap()) * u128::from(state.groups[group].source_bytes)
            / u128::from(total)) as u64;
        let charge = bytes.len() as u64 + std::mem::size_of::<Record>() as u64 + 2 * std::mem::size_of::<usize>() as u64;
        if charge > quota { return false; }
        let incoming_resident = state.tensors[tensor].entries[expert].device_resident;
        while state.groups[group].allocated.load(Ordering::Acquire).saturating_add(charge) > quota {
            let victim = state.tensors.iter().enumerate().filter(|(_, tensor)| tensor.group == group)
                .flat_map(|(ti, tensor)| tensor.entries.iter().enumerate().filter_map(move |(ei, entry)| {
                    entry.record.as_ref().filter(|record| Arc::strong_count(record) == 1
                        && (!incoming_resident || entry.device_resident))
                        .map(|record| (!entry.device_resident, record.bytes.capacity() < bytes.len(), entry.stamp, ti, ei))
                })).min();
            let Some((_, _, _, ti, ei)) = victim else { return false; };
            let mut record = state.tensors[ti].entries[ei].record.take().expect("cache victim");
            if let Some(storage) = Arc::get_mut(&mut record).filter(|record| record.bytes.capacity() >= bytes.len()) {
                // This is already claimed physical storage. Replacing an
                // unleased record neither grows RAM nor needs another host
                // observation or allocation.
                storage.bytes.resize(bytes.len(), 0);
                storage.bytes.copy_from_slice(bytes);
                state.clock += 1;
                let stamp = state.clock;
                let entry = &mut state.tensors[tensor].entries[expert];
                entry.record = Some(record);
                entry.stamp = stamp;
                return true;
            }
            drop(record);
        }
        let Some(claim) = self.claim(&mut state, charge) else {
            Self::clear_records(&mut state); return false;
        };
        let mut copy = Vec::new();
        if copy.try_reserve_exact(bytes.len()).is_err() {
            let _ = state.heap.cancel_claim(claim); return false;
        }
        copy.extend_from_slice(bytes);
        self.allocated.fetch_add(charge, Ordering::AcqRel);
        state.groups[group].allocated.fetch_add(charge, Ordering::AcqRel);
        let record = Arc::new(Record { bytes: copy, charge, allocated: self.allocated.clone(),
            group: state.groups[group].allocated.clone() });
        state.clock += 1;
        let stamp = state.clock;
        let entry = &mut state.tensors[tensor].entries[expert];
        entry.record = Some(record);
        entry.stamp = stamp;
        state.heap.cancel_claim(claim).expect("completed host cache claim");
        true
    }
}
