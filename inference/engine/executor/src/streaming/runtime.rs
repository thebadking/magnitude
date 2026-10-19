//! The host side of a streamed load's steps (spec 4.3–4.9): the routed
//! layers' directory words, the staging pools, and the watcher thread that
//! serves requests, hints and scratch fills while steps run.
//!
//! The device side of each routed layer, per bank-served step:
//!
//! ```text
//! route, resolve, hints (layers l+1, l+2) → fill: park, convert the absent
//!       → resident gate/up and output over the bound slots
//! ```
//!
//! Every step's routed entries run in submission order on one queue, which
//! gives the watcher a clock: a hint ring of layer `l` carrying step
//! sequence `q` proves every entry of earlier layers of step `q` (and of
//! every earlier step) has run, including their converts.

use super::shared_directory::{Clock, SharedDirectory};
use super::source::{ExpertSource, Outcome, Priority, Range, SourceTiming, Ticket};
use super::words::{SharedBytes, SharedWords};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Words of a layer's request region (`routed_route_streamed` in routed.seismic).
pub(crate) const REQUEST: usize = 0;
pub(crate) const ACKNOWLEDGED: usize = 1;
pub(crate) const FAILURE: usize = 2;
pub(crate) const COUNT: usize = 3;
pub(crate) const EXPERTS: usize = 4;

/// Words of a layer's scratch request region (`routed_scratch_request`):
/// the request and failure words as above; each round's acknowledgement and
/// arrival follow the header.
const SCRATCH_HEADER: usize = 4;

/// The routed tensors of an expert record, in convert order.
pub(crate) const KINDS: usize = 3;

/// Maximum idle interval of source reads, and maximum device-handshake wait.
pub(crate) const PARK_BOUND: Duration = Duration::from_secs(1);

/// One routed tensor of a layer as the model file stores it: expert `e`'s
/// bytes are `[base + e * bytes, + bytes)`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoredTensor {
    pub file: usize,
    pub base: u64,
    pub bytes: u64,
    /// The staging pool records of this tensor go to.
    pub pool: usize,
}

/// One staging pool of one (kind, source element, matrix shape), its
/// records `record_bytes` long and host-addressed: its own `0..own`, then
/// the conveyor's `own..own + conveyor` (spec 4.9), on bank records a prompt
/// borrows when a bank group lends them.
pub(crate) struct StagingPool {
    own_bytes: SharedBytes,
    conveyor_bytes: SharedBytes,
    pub record_bytes: u64,
    own: u32,
    /// Free records of each shelf, longest free first: a free record keeps
    /// the stored bytes it last held until it is taken again.
    free: [VecDeque<u32>; 2],
    /// Per record, the (layer, expert) whose stored bytes a free record
    /// still holds.
    holds: Vec<Option<(usize, i32)>>,
    /// Free own records a hint leaves (spec 4.7).
    hint_keep: usize,
    /// A CUDA view into the source arena shared by this tensor kind's
    /// encodings. Its ranges are reused only at device publication gates.
    device: Option<seismic::Tensor>,
}

/// Which records of a staging pool a read takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shelf {
    /// The pool's own: demand reads and a decode step's hints.
    Own = 0,
    /// The conveyor's (spec 4.9).
    Conveyor = 1,
}

/// What a prefetch may take of a staging pool.
#[derive(Clone, Copy)]
enum Window {
    /// A decode step's hints: at most the hint records of the pool's own.
    Hint,
    /// The conveyor: its records.
    Conveyor,
}

impl StagingPool {
    /// `own` records of `own_bytes`, of which hints may hold `hints`, and
    /// `conveyor` of `conveyor_bytes`.
    pub(crate) fn new(
        own_bytes: SharedBytes,
        conveyor_bytes: SharedBytes,
        own: u32,
        hints: u32,
        conveyor: u32,
        record_bytes: u64,
    ) -> Self {
        Self {
            own_bytes,
            conveyor_bytes,
            record_bytes,
            own,
            free: [(0..own).collect(), (own..own + conveyor).collect()],
            holds: vec![None; (own + conveyor) as usize],
            hint_keep: own.saturating_sub(hints) as usize,
            device: None,
        }
    }

    pub(crate) fn with_device(mut self, device: seismic::Tensor) -> Self {
        self.device = Some(device);
        self
    }

    /// The source ticket has finished and the destination slot's earlier
    /// converter is past its arrival gate. No readiness has been published
    /// for the new converter yet.
    fn copy_to_device(&self, source: u32, destination: usize, bytes: u64) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("missing CUDA source window")?;
        let count =
            usize::try_from(bytes).map_err(|_| "source record exceeds host address range")?;
        let offset = (destination as u64)
            .checked_mul(self.record_bytes)
            .ok_or("device source offset overflow")?;
        // SAFETY: the completed ticket owns these host bytes; the caller
        // owns this device slot until it publishes the converter's gate.
        let source = unsafe { std::slice::from_raw_parts(self.at(source, bytes), count) };
        unsafe { device.write_in_protocol(offset, source) }.map_err(|error| error.to_string())
    }

    fn shelf(&self, record: u32) -> Shelf {
        if record < self.own {
            Shelf::Own
        } else {
            Shelf::Conveyor
        }
    }

    /// The host address of record `record`, `bytes` of it in bounds.
    fn at(&self, record: u32, bytes: u64) -> *mut u8 {
        match self.shelf(record) {
            Shelf::Own => self
                .own_bytes
                .at(u64::from(record) * self.record_bytes, bytes),
            Shelf::Conveyor => self
                .conveyor_bytes
                .at(u64::from(record - self.own) * self.record_bytes, bytes),
        }
    }
}

/// The bank records the conveyor lies on (spec 4.9): group `group`'s
/// `slots`, lent while a prompt runs.
pub(crate) struct Loan {
    group: usize,
    slots: std::ops::Range<usize>,
    lent: bool,
}

impl Loan {
    pub(crate) fn new(group: usize, slots: std::ops::Range<usize>) -> Self {
        Self {
            group,
            slots,
            lent: false,
        }
    }
}

/// A routed layer's device words and its host directory.
pub(crate) struct StreamedLayer {
    pub cpu: Option<CpuStreamedLayer>,
    pub slot_map: SharedWords,
    /// Per slot of the layer's bank group, the step that last bound it.
    pub stamps: SharedWords,
    /// Per slot, the step whose conversion into it the device committed.
    pub commits: SharedWords,
    pub control: SharedWords,
    /// [5, P]: per absent choice of a request, its slot, its gate, up and
    /// down staging records (or records of -1 when the slot already holds
    /// the expert) and its acknowledgement (`routed_bank_fill`).
    pub placement: SharedWords,
    /// The layer's scratch request region (spec 4.9): every layer of a
    /// scratch-served step requests under the same sequence, so each has
    /// its own.
    pub scratch_control: SharedWords,
    /// Hints for this layer made one and two layers earlier.
    pub rings: [SharedWords; 2],
    pub stored: [StoredTensor; KINDS],
    /// Host copy: expert → slot, slot → expert.
    slot_of: Vec<i32>,
    expert_of: Vec<i32>,
    /// The last request sequence served and ring sequences read.
    served: i32,
    scratch_served: i32,
    read: [i32; 2],
}

impl StreamedLayer {
    pub(crate) fn new(
        slot_map: SharedWords,
        stamps: SharedWords,
        commits: SharedWords,
        control: SharedWords,
        placement: SharedWords,
        scratch_control: SharedWords,
        rings: [SharedWords; 2],
        stored: [StoredTensor; KINDS],
    ) -> Self {
        let experts = slot_map.len();
        let slots = stamps.len();
        // A bank starts empty (rule 5).
        slot_map.fill(-1);
        stamps.fill(-1);
        control.fill(0);
        placement.fill(0);
        scratch_control.fill(0);
        for ring in &rings {
            ring.fill(0);
        }
        Self {
            cpu: None,
            slot_map,
            stamps,
            commits,
            control,
            placement,
            scratch_control,
            rings,
            stored,
            slot_of: vec![-1; experts],
            expert_of: vec![-1; slots],
            served: 0,
            scratch_served: 0,
            read: [0; 2],
        }
    }
}

#[derive(Clone)]
pub(crate) struct CpuStreamedLayer {
    pub service: Arc<super::cpu_service::CpuExpertService>,
    pub layer: super::cpu_service::CpuExpertLayerContext,
    pub exchange: Arc<super::cpu_exchange::CpuExpertExchange>,
    pub selected: usize,
}

/// The host's heartbeat word: advanced while the watcher serves submitted
/// steps, including while it waits on reads, so a parked device wait tells a
/// slow host from a dead one (`metal/lib/streaming/control.h`).
pub(crate) struct Heartbeat {
    word: SharedWords,
    beat: i32,
    last: Instant,
}

/// How often the heartbeat advances: far inside a parked wait's patience,
/// rare enough to cost the memory system nothing.
const BEAT: Duration = Duration::from_micros(500);

impl Heartbeat {
    pub(crate) fn new(word: SharedWords) -> Self {
        word.fill(0);
        Self {
            word,
            beat: 0,
            last: Instant::now(),
        }
    }

    fn beat(&mut self) {
        if self.last.elapsed() >= BEAT {
            self.beat = self.beat.wrapping_add(1);
            self.word.store(0, self.beat);
            self.last = Instant::now();
        }
    }
}

/// How long the watcher rests after a pass that found nothing: a park
/// waits at most this long to be noticed, and the watcher does not spin on
/// memory the device is using.
const IDLE: Duration = Duration::from_micros(20);

/// Wait while source ranges make progress, keeping the device heartbeat live.
/// A batch may take longer than PARK_BOUND; only a full idle interval fails it.
fn wait_ticket(ticket: &Ticket, heartbeat: &mut Heartbeat) -> Option<Outcome> {
    let mut deadline = Instant::now() + PARK_BOUND;
    loop {
        heartbeat.beat();
        deadline = deadline.max(ticket.progress_at() + PARK_BOUND);
        let slice = (Instant::now() + Duration::from_millis(1)).min(deadline);
        if let Some(outcome) = ticket.wait_until(slice) {
            return Some(outcome);
        }
        if Instant::now() >= deadline {
            return None;
        }
    }
}

/// The load's scratch fill words (spec 4.9): each round's convert lists,
/// the copy lists and the seed list, which one layer's fill uses at a time.
pub(crate) struct ScratchFill {
    /// A flag per expert the launch routes to.
    pub flags: SharedWords,
    pub lists: Vec<[SharedWords; KINDS]>,
    pub copies: [SharedWords; KINDS],
    /// The (expert, bank slot) pairs the fill converts and keeps in the bank.
    pub seeds: SharedWords,
    experts: usize,
    /// Records staged per round.
    round_records: usize,
    /// The layer whose experts the scratch banks hold, and how many choices of
    /// the group routed to each (0 for one they do not hold). Only fills write
    /// the scratch banks and the device runs them in request order, so a later
    /// fill of the same layer (the next step of a layer-major group) reads only
    /// the experts the banks lack.
    holds: Option<(usize, Vec<u32>)>,
}

impl ScratchFill {
    pub(crate) fn new(
        flags: SharedWords,
        lists: Vec<[SharedWords; KINDS]>,
        copies: [SharedWords; KINDS],
        seeds: SharedWords,
        experts: usize,
        round_records: usize,
    ) -> Self {
        flags.fill(0);
        for list in lists
            .iter()
            .flatten()
            .chain(&copies)
            .chain(std::iter::once(&seeds))
        {
            list.fill(0);
        }
        Self {
            flags,
            lists,
            copies,
            seeds,
            experts,
            round_records,
            holds: None,
        }
    }

    fn rounds(&self) -> usize {
        self.lists.len()
    }

    fn acknowledgement(&self, round: usize) -> usize {
        SCRATCH_HEADER + round
    }

    fn arrival(&self, round: usize) -> usize {
        SCRATCH_HEADER + self.rounds() + round
    }
}

/// A fetched or fetching record: one staging record in each of the layer's
/// pools.
struct Staged {
    records: [u32; KINDS],
    ticket: Arc<Ticket>,
    /// The step sequence and layer whose resolve it was fetched for; a
    /// prefetch the resolve did not use is released once the device is past
    /// that layer.
    sequence: i32,
}

#[derive(Clone, Copy)]
enum SourcePath {
    Bank = 0,
    Scratch = 1,
}

#[derive(Clone, Copy, Default)]
struct UploadTrace {
    calls: u64,
    requested_bytes: u64,
    host_service: Duration,
}

#[derive(Default)]
struct StepTrace {
    requests: usize,
    choices: usize,
    misses: usize,
    staged: usize,
    ready: usize,
    source_bytes: [u64; 2],
    revived: usize,
    host_service: Duration,
    scratch_requests: usize,
    scratch_held: usize,
    scratch_misses: usize,
    source_tickets: [usize; 2],
    source_timing: [SourceTiming; 2],
    uploads: [[UploadTrace; KINDS]; 2],
    bank_poll_sleep: Duration,
    scratch_ticket_wait: Duration,
}

/// A count of decode steps using an expert, halved every 64 bank sequences.
/// GPU stamps expose the last use, so repeated observations count it once.
#[derive(Clone, Copy, Default)]
struct DecodeReuse {
    last_sequence: i32,
    uses: u32,
}

impl DecodeReuse {
    fn score(self, sequence: i32) -> u32 {
        let elapsed =
            (sequence.max(0) as u32 / 64).saturating_sub(self.last_sequence.max(0) as u32 / 64);
        self.uses.checked_shr(elapsed).unwrap_or(0)
    }

    fn observe(&mut self, sequence: i32) {
        if sequence > self.last_sequence {
            self.uses = self.score(sequence).saturating_add(1);
            self.last_sequence = sequence;
        }
    }
}

struct ReuseHistory {
    /// Bounded by the admitted layer/expert inventory; no weight payloads.
    experts: Vec<Vec<DecodeReuse>>,
    /// Small-row members after each layer's last scratch member. Entries
    /// live only through their physical completion, including lookahead.
    steps: std::collections::BTreeMap<i32, Vec<bool>>,
}

impl ReuseHistory {
    fn new(layers: &[StreamedLayer]) -> Self {
        Self {
            experts: layers
                .iter()
                .map(|layer| vec![DecodeReuse::default(); layer.slot_of.len()])
                .collect(),
            steps: Default::default(),
        }
    }

    fn observe(&mut self, layer: usize, expert: usize, sequence: i32) {
        if self
            .steps
            .get(&sequence)
            .is_some_and(|classes| classes[layer])
        {
            self.experts[layer][expert].observe(sequence);
        }
    }
}

struct State {
    directory: SharedDirectory,
    /// Steps a shared-cache invalidation failed.
    failed_steps: std::collections::BTreeSet<i32>,
    fetch_failures: HashMap<i32, String>,
    layers: Vec<StreamedLayer>,
    scratch: ScratchFill,
    heartbeat: Heartbeat,
    pools: Vec<StagingPool>,
    /// (layer, expert) → its staged or fetching record.
    staged: HashMap<(usize, i32), Staged>,
    /// Records converted at (sequence, layer), with their expert: free once
    /// the device is past.
    converted: Vec<(i32, usize, [u32; KINDS], i32)>,
    /// (layer, expert) → free staging records still holding its stored
    /// bytes.
    kept: HashMap<(usize, i32), [u32; KINDS]>,
    /// The first sequence of each submitted layer-major group, by the
    /// sequence of each of its later steps; any other step is its own group.
    /// Steps before the progress's group leave it.
    groups: std::collections::BTreeMap<i32, i32>,
    /// (sequence, local layer) at the last scratch member of each group.
    /// Later decode members cannot seed from the scratch banks.
    seed_steps: std::collections::BTreeSet<(i32, usize)>,
    /// The device's progress: the newest point a hint or request proved.
    progress: Clock,
    /// The bank records the conveyor lies on, when a bank group lends them.
    loan: Option<Loan>,
    /// The newest scratch-served step: a prompt holds the loan until the
    /// device is past its group.
    prompt: i32,
    /// Diagnostic observations only; absent on the ordinary serving path.
    trace: Option<HashMap<i32, StepTrace>>,
    /// Request identities for bounded cache-policy replay; disabled in serving.
    request_trace: bool,
    cpu_trace: bool,
    request_ordinal: u64,
    reuse: Option<ReuseHistory>,
    reuse_seeds: bool,
    reuse_replacement: bool,
}

impl State {
    fn observe_reuse(&mut self) {
        let Self {
            directory,
            layers,
            reuse,
            ..
        } = self;
        if let Some(history) = reuse {
            directory.visit_ready(|representative, slot, key| {
                history.observe(
                    key.layer.0 as usize,
                    key.expert.0 as usize,
                    layers[representative].stamps.load(slot),
                );
            });
        }
    }

    fn device_staging(&self) -> bool {
        self.pools.first().is_some_and(|pool| pool.device.is_some())
    }

    /// Observe the existing blocking upload without changing its publication
    /// or completion edge. The timer is host wall time, not CUDA device time.
    fn upload_source(
        &mut self,
        layer: usize,
        sequence: i32,
        path: SourcePath,
        kind: usize,
        source: u32,
        destination: usize,
    ) -> Result<(), String> {
        let tensor = self.layers[layer].stored[kind];
        let began = self.trace.as_ref().map(|_| Instant::now());
        let result = self.pools[tensor.pool].copy_to_device(source, destination, tensor.bytes);
        let elapsed = began.map(|began| began.elapsed());
        if let Some(elapsed) = elapsed {
            let sample = &mut self
                .trace
                .as_mut()
                .expect("enabled trace")
                .entry(sequence)
                .or_default()
                .uploads[path as usize][kind];
            sample.calls += 1;
            sample.requested_bytes += tensor.bytes;
            sample.host_service += elapsed;
        }
        result
    }

    fn record_source(&mut self, sequence: i32, path: SourcePath, ticket: &Ticket) {
        if let Some(steps) = &mut self.trace {
            if let Some(timing) = ticket.take_timing() {
                let trace = steps.entry(sequence).or_default();
                trace.source_tickets[path as usize] += 1;
                trace.source_timing[path as usize].add(timing);
            }
        }
    }
    /// The device point of step `sequence` at `layer`.
    fn clock(&self, sequence: i32, layer: usize) -> Clock {
        clock(&self.groups, sequence, layer)
    }

    /// The last step of `sequence`'s group: the step after which the device
    /// is past every visit of the group to a layer.
    fn last_of_group(&self, sequence: i32) -> i32 {
        let group = self.clock(sequence, 0).group;
        self.groups
            .range(sequence + 1..)
            .take_while(|&(_, &first)| first == group)
            .last()
            .map_or(sequence, |(&last, _)| last)
    }
}

fn clock(groups: &std::collections::BTreeMap<i32, i32>, sequence: i32, layer: usize) -> Clock {
    Clock {
        group: groups.get(&sequence).copied().unwrap_or(sequence),
        layer,
        sequence,
    }
}

/// The watcher's shared state and lifecycle.
pub(crate) struct StreamingRuntime {
    // Drop joins source workers before releasing their staging destinations.
    source: Arc<ExpertSource>,
    state: Arc<Mutex<State>>,
    in_flight: Arc<(AtomicUsize, Mutex<()>, Condvar)>,
    stopping: Arc<AtomicBool>,
    watcher: Option<JoinHandle<()>>,
}

impl StreamingRuntime {
    pub(crate) fn new(
        layers: Vec<StreamedLayer>,
        scratch: ScratchFill,
        heartbeat: Heartbeat,
        pools: Vec<StagingPool>,
        source: ExpertSource,
        directory: SharedDirectory,
        loan: Option<Loan>,
    ) -> Result<Self, String> {
        let sizes = layers
            .iter()
            .map(|l| (l.stamps.len(), l.commits.len()))
            .collect::<Vec<_>>();
        directory
            .validate_ports(&sizes)
            .map_err(|e| format!("shared directory ports: {e:?}"))?;
        let reuse_seeds = match std::env::var("MAGNITUDE_EXPERT_SEED_POLICY").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("prompt") => false,
            Ok("decode-reuse") => true,
            _ => return Err("MAGNITUDE_EXPERT_SEED_POLICY must be prompt or decode-reuse".into()),
        };
        let reuse_replacement =
            match std::env::var("MAGNITUDE_EXPERT_REPLACEMENT_POLICY").as_deref() {
                Err(std::env::VarError::NotPresent) | Ok("lru") => false,
                Ok("decode-reuse") => true,
                _ => {
                    return Err(
                        "MAGNITUDE_EXPERT_REPLACEMENT_POLICY must be lru or decode-reuse".into(),
                    );
                }
            };
        let reuse = (reuse_seeds || reuse_replacement).then(|| ReuseHistory::new(&layers));
        if let Some(history) = &reuse {
            let bytes = history
                .experts
                .iter()
                .map(|experts| experts.len() * std::mem::size_of::<DecodeReuse>())
                .sum::<usize>();
            eprintln!(
                "expert reuse seed={reuse_seeds} replacement={reuse_replacement} expert_counter_bytes={bytes} decay_sequences=64"
            );
        }
        let state = Arc::new(Mutex::new(State {
            directory,
            failed_steps: Default::default(),
            fetch_failures: HashMap::new(),
            layers,
            scratch,
            heartbeat,
            pools,
            staged: HashMap::new(),
            converted: Vec::new(),
            kept: HashMap::new(),
            groups: Default::default(),
            seed_steps: Default::default(),
            progress: Clock {
                group: 0,
                layer: 0,
                sequence: 0,
            },
            loan,
            prompt: 0,
            trace: std::env::var_os("MAGNITUDE_TRACE_EXPERT_STREAMING")
                .is_some()
                .then(HashMap::new),
            request_trace: std::env::var_os("MAGNITUDE_TRACE_EXPERT_REQUESTS").is_some(),
            request_ordinal: 0,
            cpu_trace: std::env::var_os("MAGNITUDE_TRACE_CPU_EXPERTS").is_some(),
            reuse,
            reuse_seeds,
            reuse_replacement,
        }));
        let source = Arc::new(source);
        let in_flight = Arc::new((AtomicUsize::new(0), Mutex::new(()), Condvar::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let watcher = {
            let (state, source, in_flight, stopping) = (
                Arc::clone(&state),
                Arc::clone(&source),
                Arc::clone(&in_flight),
                Arc::clone(&stopping),
            );
            std::thread::Builder::new()
                .name("expert-watcher".into())
                .spawn(move || watch(&state, &source, &in_flight, &stopping))
                .map_err(|error| format!("expert watcher: {error}"))?
        };
        Ok(Self {
            state,
            source,
            in_flight,
            stopping,
            watcher: Some(watcher),
        })
    }

    /// Match request-local layer indices to their exact source inventory.
    pub(crate) fn trace_inventory(
        &self,
        device: &seismic::Device,
        plan: &crate::ExpertBankPlan,
        paths: &[std::path::PathBuf],
    ) {
        let state = self.state.lock().expect("streaming state");
        if !state.request_trace {
            return;
        }
        let layers = plan
            .layers
            .iter()
            .zip(&state.layers)
            .enumerate()
            .map(|(index, (planned, layer))| {
                let sources = layer.stored.iter().enumerate().map(|(kind, stored)| {
                let pool = &plan.pools[stored.pool];
                // A physical footprint for replay, not CPU execution admission.
                let cpu_name = match pool.source.name() {
                    "gguf_q4_k" => Some("q4k@rows8"),
                    "gguf_q5_k" => Some("q5k@rows8"),
                    "gguf_q5_0" => Some("q5g32s@rows8"),
                    "gguf_q5_1" => Some("q5g32@rows8"),
                    "gguf_q8_0" => Some("q8g32s@rows8"),
                    _ => None,
                };
                let cpu_bytes = cpu_name.and_then(seismic::Element::named).and_then(|element|
                    element.canonical_byte_len(&[1, pool.matrix[0], pool.matrix[1]]).ok());
                serde_json::json!({"kind": kind, "path": paths[stored.file], "offset": stored.base,
                    "record_bytes": stored.bytes, "source_element": pool.source.name(),
                    "matrix": pool.matrix, "cpu_rows8_element": cpu_name,
                    "cpu_rows8_record_bytes": cpu_bytes})
            }).collect::<Vec<_>>();
                serde_json::json!({"layer": index, "scope": format!("{:?}", planned.scope),
                "experts": plan.experts, "selected": plan.selected, "sources": sources})
            })
            .collect::<Vec<_>>();
        eprintln!(
            "expert request {}",
            serde_json::json!({"kind": "inventory", "schema": 1,
            "bank": format!("{:p}", Arc::as_ptr(&self.source)), "device": device.info().selector,
            "layers": layers, "scope": "Miss-bearing bank requests only; all-hit and scratch frames are absent"})
        );
    }

    /// Steps `first..first + count` are about to be submitted, as one
    /// layer-major group when more than one: the watcher polls until every
    /// submitted step has completed. Called before any of their launches is
    /// committed. `seeds` supplies each local layer's last scratch sequence.
    pub(crate) fn steps_submitted(
        &self,
        first: i32,
        count: i32,
        seeds: impl IntoIterator<Item = Option<i32>>,
    ) {
        {
            let mut state = self.state.lock().expect("streaming state");
            if count > 1 {
                state
                    .groups
                    .extend((first + 1..first + count).map(|sequence| (sequence, first)));
            }
            let seeds = seeds.into_iter().collect::<Vec<_>>();
            assert_eq!(
                seeds.len(),
                state.layers.len(),
                "one seed boundary per local layer",
            );
            if let Some(history) = &mut state.reuse {
                for sequence in first..first + count {
                    history.steps.insert(
                        sequence,
                        seeds
                            .iter()
                            .map(|last| last.is_none_or(|last| sequence > last))
                            .collect(),
                    );
                }
            }
            for (layer, sequence) in seeds.into_iter().enumerate() {
                if let Some(sequence) = sequence {
                    assert!((first..first + count).contains(&sequence));
                    state.seed_steps.insert((sequence, layer));
                }
            }
        }
        let (in_flight, lock, wake) = &*self.in_flight;
        let _guard = lock.lock().expect("watcher lock");
        in_flight.fetch_add(count as usize, Ordering::AcqRel);
        wake.notify_all();
    }

    /// A step completed. Its failure, if the watcher failed any of its
    /// requests, is returned and cleared (rule 6).
    pub(crate) fn step_completed(&self, sequence: i32, fenced: bool) -> Result<(), String> {
        self.in_flight.0.fetch_sub(1, Ordering::AcqRel);
        let mut state = self.state.lock().expect("streaming state");
        if !fenced {
            if state.request_trace {
                eprintln!(
                    "expert request {}",
                    serde_json::json!({"kind": "complete", "schema": 1,
                    "bank": format!("{:p}", Arc::as_ptr(&self.source)), "sequence": sequence,
                    "fenced": false, "failed": true})
                );
            }
            if let Some(history) = &mut state.reuse {
                history.steps.remove(&sequence);
            }
            return Err("shared-cache step dropped without a completion fence".into());
        }
        let invalid = {
            let State {
                directory, layers, ..
            } = &mut *state;
            directory
                .retire_step(sequence, |layer, slot| layers[layer].commits.load(slot))
                .map_err(|e| format!("shared-cache completion: {e:?}"))?
        };
        invalidate(&mut state, &self.source, invalid);
        // Include all-hit steps and the last layers after the final miss.
        // Ready identities have conversion evidence; future observed uses
        // remain cache effects even if their lookahead is later orphaned.
        state.observe_reuse();
        if let Some(history) = &mut state.reuse {
            history.steps.remove(&sequence);
        }
        let mut failed = state.failed_steps.remove(&sequence);
        let mut device_failures = Vec::new();
        for (layer, ports) in state.layers.iter().enumerate() {
            for (kind, control) in [
                ("decode", &ports.control),
                ("scratch", &ports.scratch_control),
            ] {
                if control.load(FAILURE) == sequence {
                    failed = true;
                    device_failures.push(format!("layer {layer} {kind} device handshake"));
                    control.store(FAILURE, 0);
                }
            }
        }
        // Everything converted up to this step is in its slots now: a step
        // completes after every layer of its group.
        let done = state.clock(sequence, usize::MAX);
        if done > state.progress {
            state.progress = done;
        }
        release_past(&mut state, &self.source);
        if state.request_trace {
            eprintln!(
                "expert request {}",
                serde_json::json!({"kind": "complete", "schema": 1,
                "bank": format!("{:p}", Arc::as_ptr(&self.source)), "sequence": sequence,
                "fenced": true, "failed": failed})
            );
        }
        if state.cpu_trace {
            if let Some(cpu) = state.layers.iter().find_map(|layer| layer.cpu.as_ref()) {
                eprintln!(
                    "CPU experts {}",
                    serde_json::json!({"service":format!("{:p}",Arc::as_ptr(&cpu.service)),
                    "sequence":sequence,"fenced":true,"snapshot":cpu.service.snapshot()})
                );
            }
        }
        if let Some(trace) = state
            .trace
            .as_mut()
            .and_then(|steps| steps.remove(&sequence))
        {
            let (cache_hit_bytes, file_read_bytes) = self.source.counters();
            eprintln!(
                "expert source bank={:p} sequence={sequence} cache_hit_bytes={cache_hit_bytes} file_read_bytes={file_read_bytes}",
                Arc::as_ptr(&self.source)
            );
            eprintln!(
                "expert streaming bank={:p} sequence={sequence} requests={} choices={} misses={} staged={} ready={} demand_bytes={} prefetch_bytes={} revived={} host_service_ms={:.3} scratch_requests={} scratch_held={} scratch_misses={}",
                Arc::as_ptr(&self.source),
                trace.requests,
                trace.choices,
                trace.misses,
                trace.staged,
                trace.ready,
                trace.source_bytes[0],
                trace.source_bytes[1],
                trace.revived,
                trace.host_service.as_secs_f64() * 1000.0,
                trace.scratch_requests,
                trace.scratch_held,
                trace.scratch_misses,
            );
            for (path, name) in ["bank", "scratch"].into_iter().enumerate() {
                let source = trace.source_timing[path];
                let uploads = &trace.uploads[path];
                let calls = uploads.iter().map(|sample| sample.calls).sum::<u64>();
                let bytes = uploads
                    .iter()
                    .map(|sample| sample.requested_bytes)
                    .sum::<u64>();
                let service = uploads
                    .iter()
                    .map(|sample| sample.host_service)
                    .sum::<Duration>();
                eprintln!(
                    "expert service bank={:p} sequence={sequence} path={name} source_tickets={} source_ranges={} source_first_start_ms_sum={:.3} source_worker_ms_sum={:.3} source_span_ms_sum={:.3} upload_calls={calls} upload_requested_bytes={bytes} upload_host_ms={:.3} gate_upload_ms={:.3} up_upload_ms={:.3} down_upload_ms={:.3} bank_poll_sleep_ms={:.3} scratch_ticket_wait_ms={:.3}",
                    Arc::as_ptr(&self.source),
                    trace.source_tickets[path],
                    source.ranges,
                    source.first_start_delay.as_secs_f64() * 1000.0,
                    source.worker_service.as_secs_f64() * 1000.0,
                    source.completion_span.as_secs_f64() * 1000.0,
                    service.as_secs_f64() * 1000.0,
                    uploads[0].host_service.as_secs_f64() * 1000.0,
                    uploads[1].host_service.as_secs_f64() * 1000.0,
                    uploads[2].host_service.as_secs_f64() * 1000.0,
                    if path == 0 {
                        trace.bank_poll_sleep.as_secs_f64() * 1000.0
                    } else {
                        0.0
                    },
                    if path == 1 {
                        trace.scratch_ticket_wait.as_secs_f64() * 1000.0
                    } else {
                        0.0
                    },
                );
            }
        }
        let detail = state.fetch_failures.remove(&sequence);
        if failed {
            return Err(format!(
                "step {sequence}: {}",
                detail.unwrap_or_else(|| if device_failures.is_empty() {
                    "shared-cache invalidation failed".into()
                } else {
                    device_failures.join("; ")
                })
            ));
        }
        Ok(())
    }
}

impl Drop for StreamingRuntime {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.in_flight.2.notify_all();
        if let Some(watcher) = self.watcher.take() {
            let _ = watcher.join();
        }
    }
}

fn watch(
    state: &Mutex<State>,
    source: &ExpertSource,
    in_flight: &(AtomicUsize, Mutex<()>, Condvar),
    stopping: &AtomicBool,
) {
    let (count, lock, wake) = in_flight;
    loop {
        if stopping.load(Ordering::Acquire) {
            return;
        }
        if count.load(Ordering::Acquire) == 0 {
            let guard = lock.lock().expect("watcher lock");
            if count.load(Ordering::Acquire) == 0 && !stopping.load(Ordering::Acquire) {
                drop(wake.wait_timeout(guard, Duration::from_millis(50)));
            }
            continue;
        }
        let mut state = state.lock().expect("streaming state");
        state.heartbeat.beat();
        let progressed = poll(&mut state, source);
        drop(state);
        if !progressed {
            std::thread::sleep(IDLE);
        }
    }
}

/// One pass over every layer: new hints first (they also advance the clock),
/// then requests. True when it found anything.
fn poll(state: &mut State, source: &ExpertSource) -> bool {
    fence(Ordering::SeqCst);
    let mut progressed = false;
    let layers = state.layers.len();
    for layer in 0..layers {
        for distance in 0..2 {
            // Layer `l`'s ring `d` holds its hints for layer `l + d + 1`.
            let target = layer + distance + 1;
            if target >= layers {
                continue;
            }
            let sequence = state.layers[layer].rings[distance].load(0);
            if sequence == state.layers[layer].read[distance] {
                continue;
            }
            state.layers[layer].read[distance] = sequence;
            progressed = true;
            // A hint of layer `l` proves the device is past layer `l - 1`
            // of this step.
            let point = state.clock(sequence, layer);
            advance(state, point, source);
            let ring = &state.layers[layer].rings[distance];
            let count = usize::try_from(ring.load(1))
                .unwrap_or(0)
                .min(ring.len() - 2);
            let experts = (0..count).map(|i| ring.load(2 + i)).collect::<Vec<_>>();
            for expert in experts {
                prefetch(state, source, target, expert, sequence, Window::Hint);
            }
        }
    }
    for layer in 0..layers {
        let sequence = state.layers[layer].control.load(REQUEST);
        if sequence != 0 && sequence != state.layers[layer].served {
            serve(state, source, layer, sequence);
            progressed = true;
        }
    }
    for layer in 0..layers {
        let sequence = state.layers[layer].scratch_control.load(REQUEST);
        if sequence != 0 && sequence != state.layers[layer].scratch_served {
            serve_scratch(state, source, layer, sequence);
            progressed = true;
        }
    }
    progressed
}

/// The device is at least at `point`: release what it no longer needs.
fn advance(state: &mut State, point: Clock, source: &ExpertSource) {
    if point > state.progress {
        let invalid = {
            let State {
                directory, layers, ..
            } = &mut *state;
            directory
                .retire_progress(point, |layer, slot| layers[layer].commits.load(slot))
                .expect("device-published shared-cache progress")
        };
        invalidate(state, source, invalid);
        state.observe_reuse();
        state.progress = point;
        release_past(state, source);
    }
}

/// Mappings the directory withdrew: unmap them and fail their steps.
fn invalidate(
    state: &mut State,
    source: &ExpertSource,
    invalid: Vec<(Clock, super::arena::Location)>,
) {
    for (clock, location) in invalid {
        state.failed_steps.insert(clock.sequence);
        state
            .fetch_failures
            .entry(clock.sequence)
            .or_insert_with(|| {
                format!(
                    "uncommitted cache fill: layer {}, expert {}, slot {}, commit {} (expected {})",
                    clock.layer,
                    location.key.expert.0,
                    location.slot,
                    state.layers[clock.layer].commits.load(location.slot),
                    clock.sequence
                )
            });
        unmap(state, source, location);
        state.layers[location.key.layer.0 as usize]
            .control
            .store(FAILURE, clock.sequence);
    }
}

/// The bank no longer holds `location`'s expert in its slot.
fn unmap(state: &mut State, source: &ExpertSource, location: super::arena::Location) {
    let layer = &mut state.layers[location.key.layer.0 as usize];
    let expert = location.key.expert.0 as usize;
    if layer.slot_of[expert] == location.slot as i32 {
        layer.slot_of[expert] = -1;
        layer.slot_map.store(expert, -1);
        layer.expert_of[location.slot] = -1;
        for tensor in layer.stored {
            source.device_residency(tensor, expert, false);
        }
    }
}

/// Borrow the conveyor's bank records for a prompt (spec 4.9): the
/// directory evicts the experts they hold, unless a step still reads or
/// fills one, when the conveyor waits for a later round. The device runs
/// steps in order and is parked at the prompt's request, so no step reads
/// an evicted expert after its map entry is cleared.
fn borrow(state: &mut State, source: &ExpertSource) -> bool {
    let Some(loan) = &state.loan else {
        return true;
    };
    if loan.lent {
        return true;
    }
    let (group, slots) = (loan.group, loan.slots.clone());
    let Ok(evicted) = state.directory.lend(group, slots) else {
        return false;
    };
    for location in evicted {
        unmap(state, source, location);
    }
    state.loan.as_mut().expect("a loan").lent = true;
    true
}

/// Give the conveyor's bank records back once the device is past the
/// prompt and none of them is taken: whatever bytes they keep are forgotten,
/// and the bank fills them as empty slots.
fn give_back(state: &mut State) {
    if !state.loan.as_ref().is_some_and(|loan| loan.lent)
        || state.clock(state.last_of_group(state.prompt), usize::MAX) > state.progress
        || state.pools.iter().any(|pool| {
            pool.free[Shelf::Conveyor as usize].len() + pool.own as usize != pool.holds.len()
        })
    {
        return;
    }
    for pool in 0..state.pools.len() {
        let own = state.pools[pool].own as usize;
        for record in own..state.pools[pool].holds.len() {
            if let Some(key) = state.pools[pool].holds[record].take() {
                forget(state, key);
            }
        }
    }
    let loan = state.loan.as_mut().expect("a loan");
    loan.lent = false;
    let (group, slots) = (loan.group, loan.slots.clone());
    state
        .directory
        .restore(group, slots)
        .expect("the conveyor's lent records");
}

/// Free converted records and cancel or free unused prefetches of layers the
/// device has passed.
fn release_past(state: &mut State, source: &ExpertSource) {
    let progress = state.progress;
    let mut freed = Vec::new();
    let State {
        converted, groups, ..
    } = &mut *state;
    converted.retain(|&(sequence, layer, records, expert)| {
        if clock(groups, sequence, layer) < progress {
            freed.push((records, layer, expert));
            false
        } else {
            true
        }
    });
    // A step before the progress's group is past at every layer, as its own
    // group or its layer-major one.
    groups.retain(|&sequence, _| sequence >= progress.group);
    state
        .seed_steps
        .retain(|&(sequence, _)| sequence >= progress.group);
    for (records, layer, expert) in freed {
        free_records(state, layer, records, (expert >= 0).then_some(expert));
    }
    let stale = state
        .staged
        .iter()
        .filter(|((layer, _), staged)| state.clock(staged.sequence, *layer) < progress)
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for key in stale {
        let staged = &state.staged[&key];
        match staged.ticket.poll() {
            None => {
                // Still reading: cancel if it has not started, else let it end
                // and release it on a later pass.
                if source.cancel(&staged.ticket) {
                    let staged = state.staged.remove(&key).expect("present");
                    free_records(state, key.0, staged.records, None);
                }
            }
            Some(outcome) => {
                let staged = state.staged.remove(&key).expect("present");
                let holds = (outcome == Outcome::Done).then_some(key.1);
                free_records(state, key.0, staged.records, holds);
            }
        }
    }
    give_back(state);
}

/// Return a record's staging records to their pools. With `holds`, they
/// still hold that expert's stored bytes: a later prefetch or request takes
/// them back without a read (`revive`) until one of them is taken for
/// another record.
fn free_records(state: &mut State, layer: usize, records: [u32; KINDS], holds: Option<i32>) {
    if let Some(expert) = holds {
        forget(state, (layer, expert));
    }
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let pool = &mut state.pools[pools[kind]];
        let shelf = pool.shelf(record);
        pool.free[shelf as usize].push_back(record);
        pool.holds[record as usize] = holds.map(|expert| (layer, expert));
    }
    if let Some(expert) = holds {
        state.kept.insert((layer, expert), records);
    }
}

/// Whether `pool` can take a record of `shelf`: the conveyor's only while
/// its bank records are lent.
fn available(state: &State, pool: usize, shelf: Shelf) -> bool {
    !state.pools[pool].free[shelf as usize].is_empty()
        && (shelf == Shelf::Own || state.loan.as_ref().is_none_or(|loan| loan.lent))
}

/// Take `pool`'s longest-free record of `shelf`, forgetting what it held.
fn take_record(state: &mut State, pool: usize, shelf: Shelf) -> Option<u32> {
    if !available(state, pool, shelf) {
        return None;
    }
    let record = state.pools[pool].free[shelf as usize].pop_front()?;
    if let Some(key) = state.pools[pool].holds[record as usize].take() {
        forget(state, key);
    }
    Some(record)
}

/// A kept record is no longer intact.
fn forget(state: &mut State, key: (usize, i32)) {
    let Some(records) = state.kept.remove(&key) else {
        return;
    };
    let pools = state.layers[key.0].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let holds = &mut state.pools[pools[kind]].holds[record as usize];
        if *holds == Some(key) {
            *holds = None;
        }
    }
}

/// Take back the staging records still holding `expert`'s stored bytes.
fn revive(state: &mut State, layer: usize, expert: i32) -> Option<[u32; KINDS]> {
    let records = state.kept.remove(&(layer, expert))?;
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    for (kind, record) in records.into_iter().enumerate() {
        let pool = &mut state.pools[pools[kind]];
        pool.holds[record as usize] = None;
        let free = &mut pool.free[pool.shelf(record) as usize];
        let index = free
            .iter()
            .position(|&free| free == record)
            .expect("a kept record is free");
        free.remove(index);
    }
    Some(records)
}

/// Reserve a staging record of `shelf` in each of the layer's pools and
/// queue its reads, or take back free records still holding the expert's
/// bytes. A decode fill converts from the pools' own records only
/// (`routed_bank_fill`), so a read for one never takes back the conveyor's.
fn fetch(
    state: &mut State,
    source: &ExpertSource,
    layer: usize,
    expert: i32,
    priority: Priority,
    shelf: Shelf,
    sequence: i32,
) -> bool {
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    let revivable = state.kept.get(&(layer, expert)).is_some_and(|records| {
        shelf == Shelf::Conveyor
            || records
                .iter()
                .zip(pools)
                .all(|(&record, pool)| state.pools[pool].shelf(record) == Shelf::Own)
    });
    if let Some(records) = revivable.then(|| revive(state, layer, expert)).flatten() {
        if let Some(steps) = &mut state.trace {
            steps.entry(sequence).or_default().revived += 1;
        }
        state.staged.insert(
            (layer, expert),
            Staged {
                records,
                ticket: Ticket::done(priority),
                sequence,
            },
        );
        return true;
    }
    let stored = state.layers[layer].stored;
    if !stored
        .iter()
        .all(|tensor| available(state, tensor.pool, shelf))
    {
        return false;
    }
    let mut records = [0u32; KINDS];
    let mut ranges = Vec::with_capacity(KINDS);
    for (kind, tensor) in stored.iter().enumerate() {
        let record = take_record(state, tensor.pool, shelf).expect("checked free");
        records[kind] = record;
        ranges.push(Range {
            file: tensor.file,
            offset: tensor.base + u64::try_from(expert).expect("an expert index") * tensor.bytes,
            bytes: usize::try_from(tensor.bytes).expect("a record fits the host"),
            destination: state.pools[tensor.pool].at(record, tensor.bytes),
        });
    }
    if let Some(steps) = &mut state.trace {
        let kind = if priority == Priority::Demand { 0 } else { 1 };
        steps.entry(sequence).or_default().source_bytes[kind] +=
            ranges.iter().map(|range| range.bytes as u64).sum::<u64>();
    }
    let ticket = source.submit(ranges, priority);
    state.staged.insert(
        (layer, expert),
        Staged {
            records,
            ticket,
            sequence,
        },
    );
    true
}

/// Decode's converter addresses only the own shelf. A prompt's conveyor
/// read may still be running when a small member of its group needs it.
/// Complete that read before moving its bytes, retaining its original
/// destination through a failure or timeout.
fn stage_for_decode(
    state: &mut State,
    source: &ExpertSource,
    layer: usize,
    expert: i32,
    sequence: i32,
) -> Result<(), String> {
    let key = (layer, expert);
    if state.device_staging() {
        state
            .staged
            .get_mut(&key)
            .expect("a pending decode source")
            .sequence = sequence;
        return Ok(());
    }
    let staged = state.staged.get(&key).expect("a pending decode source");
    let records = staged.records;
    let stored = state.layers[layer].stored;
    let own = records
        .iter()
        .zip(stored)
        .all(|(&record, tensor)| state.pools[tensor.pool].shelf(record) == Shelf::Own);
    if own {
        state.staged.get_mut(&key).unwrap().sequence = sequence;
        return Ok(());
    }
    let ticket = Arc::clone(&staged.ticket);
    source.promote(&ticket);
    if wait_ticket(&ticket, &mut state.heartbeat) != Some(Outcome::Done) {
        return Err(format!("conveyor source: {}", ticket.diagnostic()));
    }
    let mut needed = HashMap::<usize, usize>::new();
    for tensor in stored {
        *needed.entry(tensor.pool).or_default() += 1;
    }
    let available = |state: &State| {
        needed
            .iter()
            .all(|(&pool, &count)| state.pools[pool].free[Shelf::Own as usize].len() >= count)
    };
    if !available(state) {
        reclaim_prefetches(state, source, layer);
    }
    if !available(state) {
        return Err("no decode staging for a conveyor source".into());
    }
    let copied = std::array::from_fn(|kind| {
        take_record(state, stored[kind].pool, Shelf::Own).expect("reserved own capacity")
    });
    for (kind, tensor) in stored.iter().enumerate() {
        let pool = &state.pools[tensor.pool];
        let from = pool.at(records[kind], tensor.bytes);
        let to = pool.at(copied[kind], tensor.bytes);
        // The ticket completed, the old records remain reserved, and the
        // new own records are distinct reserved ranges with no device reader.
        let bytes = usize::try_from(tensor.bytes).expect("an admitted source record fits the host");
        for offset in (0..bytes).step_by(1024 * 1024) {
            state.heartbeat.beat();
            let count = (bytes - offset).min(1024 * 1024);
            unsafe {
                std::ptr::copy_nonoverlapping(from.add(offset), to.add(offset), count);
            }
        }
    }
    state.staged.insert(
        key,
        Staged {
            records: copied,
            ticket: Ticket::done(Priority::Demand),
            sequence,
        },
    );
    free_records(state, layer, records, None);
    Ok(())
}

/// A hint: prefetch the expert unless it is resident, staged or fetching.
fn prefetch(
    state: &mut State,
    source: &ExpertSource,
    layer: usize,
    expert: i32,
    sequence: i32,
    window: Window,
) {
    let Ok(index) = usize::try_from(expert) else {
        return;
    };
    if index >= state.layers[layer].slot_of.len()
        || state.layers[layer].slot_of[index] >= 0
        || state.staged.contains_key(&(layer, expert))
    {
        return;
    }
    // A hint never takes the records a demand read would need.
    let shelf = match window {
        Window::Hint => {
            let admitted = state.layers[layer].stored.iter().all(|tensor| {
                let pool = &state.pools[tensor.pool];
                pool.free[Shelf::Own as usize].len() > pool.hint_keep
            });
            if !admitted {
                return;
            }
            Shelf::Own
        }
        Window::Conveyor => Shelf::Conveyor,
    };
    fetch(
        state,
        source,
        layer,
        expert,
        Priority::Prefetch,
        shelf,
        sequence,
    );
}

/// A parked step's request (spec 4.6 step 4): resolve it against the shared
/// directory, deliver every absent expert into staging, publish each choice's
/// placement and its acknowledgement as its record arrives, then the request's.
/// CPU work has no sequence-state writes. Availability or execution failure
/// therefore selects the ordinary GPU path before any GPU slots are reserved.
fn try_cpu(
    state: &mut State,
    layer: usize,
    sequence: i32,
    choices: &[i32],
    routed: &[i32],
) -> (Option<usize>, Option<(CpuStreamedLayer, i32)>) {
    let Some(cpu) = state.layers[layer].cpu.clone() else {
        return (None, None);
    };
    if choices.len() != cpu.selected || choices.iter().filter(|&&choice| choice >= 0).count() != 1 {
        return (None, None);
    };
    let position = choices
        .iter()
        .position(|&choice| choice >= 0)
        .expect("one absent choice");
    let expert = routed[position];
    if cpu.exchange.failure().is_some() {
        return (None, None);
    }
    let deadline = Instant::now() + PARK_BOUND;
    let request = loop {
        cpu.exchange.pulse();
        state.heartbeat.beat();
        match cpu.exchange.try_request() {
            Ok(Some(request)) => break request,
            Ok(None) if Instant::now() < deadline => std::thread::yield_now(),
            _ => {
                cpu.exchange.abort();
                return (None, None);
            }
        }
    };
    if request.sequence() != sequence {
        drop(request);
        return (None, None);
    };
    let needs_conversion = !state.directory.is_ready(layer, expert);
    let mut contribution = None;
    if needs_conversion {
        cpu.service.eligible.fetch_add(1, Ordering::Relaxed);
        if let Ok(Some(lease)) = cpu.service.get(&cpu.layer, expert as usize) {
            contribution = cpu
                .service
                .execute(&cpu.layer, &lease, request.activation())
                .ok();
        }
    }
    let used_cpu = contribution.is_some();
    let contribution = contribution
        .unwrap_or_else(|| vec![0; cpu.exchange.contribution().extents()[1] as usize * 4]);
    if request.complete(&contribution).is_err() {
        return (None, None);
    };
    if used_cpu {
        // A host ACK alone does not establish successful device consumption:
        // the device can have reached its watchdog at that exact publication.
        while !cpu.exchange.consumed(sequence) {
            cpu.exchange.pulse();
            state.heartbeat.beat();
            if cpu.exchange.failure().is_some() || Instant::now() >= deadline {
                cpu.exchange.abort();
                return (None, None);
            }
            std::thread::yield_now();
        }
        cpu.service.hits.fetch_add(1, Ordering::Relaxed);
        (Some(position), None)
    } else {
        (None, needs_conversion.then_some((cpu, expert)))
    }
}

fn serve(state: &mut State, source: &ExpertSource, layer: usize, sequence: i32) {
    let started = Instant::now();
    let mut deadline = started + PARK_BOUND;
    let point = state.clock(sequence, layer);
    advance(state, point, source);
    let width = state.layers[layer].placement.len() / 5;
    let count = usize::try_from(state.layers[layer].control.load(COUNT)).unwrap_or(0);
    let experts = state.layers[layer].slot_of.len();
    let fail = |state: &mut State| {
        state.layers[layer].control.store(FAILURE, sequence);
        state.layers[layer].served = sequence;
    };
    if count == 0 || count > width {
        fail(state);
        return;
    }
    let choices = (0..count)
        .map(|i| state.layers[layer].control.load(EXPERTS + i))
        .collect::<Vec<_>>();
    let routed = choices
        .iter()
        .map(|&word| {
            if word < 0 {
                -(i64::from(word)) - 1
            } else {
                i64::from(word)
            }
        })
        .collect::<Vec<_>>();
    if routed
        .iter()
        .any(|&expert| expert < 0 || expert as usize >= experts)
    {
        fail(state);
        return;
    }
    let routed = routed.into_iter().map(|e| e as i32).collect::<Vec<_>>();
    let (cpu_choice, cpu_retain) = try_cpu(state, layer, sequence, &choices, &routed);
    let gpu_routed = match cpu_choice {
        Some(position) => std::borrow::Cow::Owned(
            routed
                .iter()
                .enumerate()
                .filter_map(|(index, &expert)| (index != position).then_some(expert))
                .collect::<Vec<_>>(),
        ),
        None => std::borrow::Cow::Borrowed(routed.as_slice()),
    };
    let original_choice =
        |index: usize| index + usize::from(cpu_choice.is_some_and(|position| index >= position));
    state
        .directory
        .sync_stamps(
            layer,
            (0..state.layers[layer].stamps.len()).map(|s| state.layers[layer].stamps.load(s)),
        )
        .expect("shared GPU stamps");
    let resolution = if state.reuse_replacement {
        // Current hits are pinned by begin before any victim is ranked.
        // The equal-sized compatible arena needs no resident-size divisor.
        let history = state.reuse.as_ref().expect("enabled reuse replacement");
        let layers = &state.layers;
        let priority = |key: super::arena::Key| {
            let layer = key.layer.0 as usize;
            let uses = history.experts[layer][key.expert.0 as usize].score(sequence);
            let source_bytes: u128 = layers[layer]
                .stored
                .iter()
                .map(|tensor| u128::from(tensor.bytes))
                .sum();
            u128::from(uses) * source_bytes
        };
        state
            .directory
            .begin_with_priority(point, &gpu_routed, Some(&priority))
    } else {
        state.directory.begin(point, &gpu_routed)
    }
    .expect("validated shared route union");
    if state.request_trace {
        state.request_ordinal += 1;
        let evicted = resolution
            .evicted
            .iter()
            .map(|location| {
                serde_json::json!({
                    "layer": location.key.layer.0, "expert": location.key.expert.0,
                })
            })
            .collect::<Vec<_>>();
        eprintln!(
            "expert request {}",
            serde_json::json!({"kind": "request", "schema": 1,
            "bank": format!("{source:p}"), "ordinal": state.request_ordinal,
            "sequence": sequence, "layer": layer, "choices": routed,
            "initial_gpu_hits": choices.iter().map(|&choice| choice < 0).collect::<Vec<_>>(),
            "needs_conversion": (0..count).map(|index|if cpu_choice==Some(index) {false} else {
                resolution.choices[index-usize::from(cpu_choice.is_some_and(|position|index>position))].needs_conversion }).collect::<Vec<_>>(),
            "cpu_choice":cpu_choice,
            "evicted": evicted})
        );
    }
    if let Some(steps) = &mut state.trace {
        let trace = steps.entry(sequence).or_default();
        trace.requests += 1;
        trace.choices += count;
        for choice in &resolution.choices {
            if !choice.needs_conversion {
                continue;
            }
            trace.misses += 1;
            if let Some(staged) = state
                .staged
                .get(&(layer, choice.location.key.expert.0 as i32))
            {
                trace.staged += 1;
                trace.ready += usize::from(staged.ticket.poll() == Some(Outcome::Done));
            }
        }
    }
    for victim in &resolution.evicted {
        unmap(state, source, *victim);
    }
    let mut pending = Vec::new();
    for choice in &resolution.choices {
        let expert = choice.location.key.expert.0 as i32;
        if !choice.needs_conversion || pending.contains(&expert) {
            continue;
        }
        pending.push(expert);
        let slot = choice.location.slot;
        state.layers[layer].commits.store(slot, 0);
        state.layers[layer].slot_of[expert as usize] = slot as i32;
        state.layers[layer].expert_of[slot] = expert;
        state.layers[layer].stamps.store(slot, sequence);
        for tensor in state.layers[layer].stored {
            source.device_residency(tensor, expert as usize, true);
        }
        match state.staged.get(&(layer, expert)) {
            Some(staged) => {
                if staged.ticket.priority() == Priority::Prefetch {
                    source.promote(&staged.ticket);
                }
            }
            None => {
                let demand = |state: &mut State| {
                    fetch(
                        state,
                        source,
                        layer,
                        expert,
                        Priority::Demand,
                        Shelf::Own,
                        sequence,
                    )
                };
                if !demand(state) {
                    reclaim_prefetches(state, source, layer);
                    if !demand(state) {
                        let point = state.clock(sequence, layer);
                        state
                            .directory
                            .submitted(point)
                            .expect("failed immutable frame");
                        fail(state);
                        return;
                    }
                }
            }
        }
    }
    for &expert in &pending {
        if let Err(detail) = stage_for_decode(state, source, layer, expert, sequence) {
            state
                .directory
                .submitted(point)
                .expect("failed immutable frame");
            state
                .fetch_failures
                .entry(sequence)
                .or_insert_with(|| format!("layer {layer}, expert {expert}: {detail}"));
            fail(state);
            return;
        }
    }
    // All slots and raw source addresses become immutable before any source
    // readiness is published. Reserved source records remain staged until
    // their tickets finish, even when another record fails mid-request.
    let device_staging = state.device_staging();
    if let Some(c) = cpu_choice {
        for row in 0..4 {
            state.layers[layer].placement.store(row * width + c, -1);
        }
        state.layers[layer].placement.store(4 * width + c, 0);
    }
    for (index, choice) in resolution.choices.iter().enumerate() {
        let c = original_choice(index);
        if choices[c] < 0 {
            continue;
        }
        state.layers[layer]
            .placement
            .store(c, choice.location.slot as i32);
        let records = if choice.needs_conversion && device_staging {
            let index = pending
                .iter()
                .position(|&expert| expert == routed[c])
                .expect("a pending conversion");
            [index as i32; KINDS]
        } else if choice.needs_conversion {
            state.staged[&(layer, routed[c])].records.map(|r| r as i32)
        } else {
            [-1; KINDS]
        };
        for (kind, record) in records.into_iter().enumerate() {
            state.layers[layer]
                .placement
                .store((1 + kind) * width + c, record);
        }
        state.layers[layer].placement.store(4 * width + c, 0);
    }
    let point = state.clock(sequence, layer);
    state
        .directory
        .submitted(point)
        .expect("immutable source publication");
    let mut outstanding = pending.clone();
    if outstanding.is_empty() {
        state.layers[layer].control.store(ACKNOWLEDGED, sequence);
    }
    if let Some(c) = cpu_choice {
        state.layers[layer].placement.store(4 * width + c, sequence);
    }
    for (index, choice) in resolution.choices.iter().enumerate() {
        let c = original_choice(index);
        if choices[c] >= 0 && !choice.needs_conversion {
            state.layers[layer].placement.store(4 * width + c, sequence);
        }
    }
    while !outstanding.is_empty() {
        state.heartbeat.beat();
        for expert in &outstanding {
            deadline =
                deadline.max(state.staged[&(layer, *expert)].ticket.progress_at() + PARK_BOUND);
        }
        if Instant::now() >= deadline || state.layers[layer].control.load(FAILURE) == sequence {
            let reason = if Instant::now() >= deadline {
                "fetch made no progress"
            } else {
                "device handshake"
            };
            let tickets = outstanding
                .iter()
                .map(|expert| {
                    format!(
                        "expert {expert}: {}",
                        state.staged[&(layer, *expert)].ticket.diagnostic()
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            state.fetch_failures.entry(sequence).or_insert_with(|| {
                format!(
                    "layer {layer} {reason} after {:?}: {tickets}",
                    started.elapsed()
                )
            });
            fail(state);
            return;
        }
        let mut arrived = false;
        let mut index = 0;
        while index < outstanding.len() {
            let expert = outstanding[index];
            match state.staged[&(layer, expert)].ticket.poll() {
                None => {
                    index += 1;
                    continue;
                }
                Some(Outcome::Done) => {}
                Some(_) => {
                    let detail = state.staged[&(layer, expert)].ticket.diagnostic();
                    state
                        .fetch_failures
                        .entry(sequence)
                        .or_insert_with(|| format!("layer {layer}, expert {expert}: {detail}"));
                    fail(state);
                    return;
                }
            }
            if device_staging {
                let records = state.staged[&(layer, expert)].records;
                let destination = pending
                    .iter()
                    .position(|&pending| pending == expert)
                    .expect("a pending conversion");
                for kind in 0..KINDS {
                    state.heartbeat.beat();
                    if let Err(error) = state.upload_source(
                        layer,
                        sequence,
                        SourcePath::Bank,
                        kind,
                        records[kind],
                        destination,
                    ) {
                        state.fetch_failures.entry(sequence).or_insert_with(|| {
                            format!(
                                "layer {layer}, expert {expert}: device source transfer: {error}"
                            )
                        });
                        fail(state);
                        return;
                    }
                }
            }
            let staged = state
                .staged
                .remove(&(layer, expert))
                .expect("completed source record");
            state.record_source(sequence, SourcePath::Bank, &staged.ticket);
            state
                .converted
                .push((sequence, layer, staged.records, expert));
            outstanding.swap_remove(index);
            // The last source flag cannot let the gate pass finish before
            // the aggregate completion seal required by output publication.
            if outstanding.is_empty() {
                state.layers[layer].control.store(ACKNOWLEDGED, sequence);
            }
            for (index, choice) in resolution.choices.iter().enumerate() {
                let c = original_choice(index);
                if choices[c] >= 0 && choice.location.key.expert.0 == expert as u32 {
                    state.layers[layer].placement.store(4 * width + c, sequence);
                }
            }
            arrived = true;
        }
        if !arrived {
            let began = state.trace.as_ref().map(|_| Instant::now());
            std::thread::sleep(IDLE);
            let elapsed = began.map(|began| began.elapsed());
            if let Some(elapsed) = elapsed {
                state
                    .trace
                    .as_mut()
                    .expect("enabled trace")
                    .entry(sequence)
                    .or_default()
                    .bank_poll_sleep += elapsed;
            }
        }
    }
    // Only reuse source records the ordinary GPU miss already fetched. CUDA's
    // uploads are complete and its converter reads the device copy. This
    // watcher still owns the host records until its next progress retirement.
    if let Some((cpu, expert)) = cpu_retain {
        if let Some(&(_, _, records, _)) = state
            .converted
            .last()
            .filter(|&&(seq, owner, _, key)| seq == sequence && owner == layer && key == expert)
        {
            let stored = state.layers[layer].stored;
            // SAFETY: the completed tickets and this watcher's exclusion keep
            // all three host ranges immutable throughout the synchronous fill.
            let raw = std::array::from_fn(|kind| unsafe {
                std::slice::from_raw_parts(
                    state.pools[stored[kind].pool].at(records[kind], stored[kind].bytes),
                    stored[kind].bytes as usize,
                )
            });
            if let Err(error) = cpu.service.retain(&cpu.layer, expert as usize, raw) {
                eprintln!("CPU expert retention bypass layer={layer} expert={expert}: {error}");
            }
        }
    }
    state.layers[layer].served = sequence;
    if let Some(steps) = &mut state.trace {
        steps.entry(sequence).or_default().host_service += started.elapsed();
    }
}

/// The experts of `layer` a prompt group seeds at its last scratch step: its most
/// routed (`routed`, the group's choices per expert) up to the layer's share,
/// those the bank lacks. To make room within the share it evicts the layer's
/// least recently read other experts, never the step's `hits`, which the fill
/// copies; a short prompt evicts no more than its seeds need.
fn keep_routed(
    state: &mut State,
    source: &ExpertSource,
    layer: usize,
    sequence: i32,
    routed: &[u32],
    hits: &[i32],
) -> Vec<i32> {
    let share = state
        .directory
        .share(layer)
        .expect("a streamed layer of the directory");
    state.observe_reuse();
    let score = |expert: usize| {
        state
            .reuse
            .as_ref()
            .filter(|_| state.reuse_seeds)
            .map_or(0, |history| history.experts[layer][expert].score(sequence))
    };
    let mut kept = (0..routed.len() as i32)
        .filter(|&expert| {
            routed[expert as usize] > 0
                || (state.layers[layer].slot_of[expert as usize] >= 0 && score(expert as usize) > 0)
        })
        .collect::<Vec<_>>();
    kept.sort_by_key(|&expert| {
        std::cmp::Reverse((score(expert as usize), routed[expert as usize]))
    });
    kept.truncate(share);
    let slot_of = &state.layers[layer].slot_of;
    let mut seeded = kept
        .iter()
        .copied()
        .filter(|&expert| slot_of[expert as usize] < 0)
        .collect::<Vec<_>>();
    // Hits remain pinned until their scratch copies finish. New seeds share
    // that frame and can use only the slots not occupied by those hits.
    let capacity = state
        .directory
        .request_capacity(layer)
        .expect("a streamed layer");
    seeded.truncate(capacity.saturating_sub(hits.len()));
    let resident = slot_of.iter().filter(|&&slot| slot >= 0).count();
    let mut others = (0..slot_of.len() as i32)
        .filter(|&expert| {
            slot_of[expert as usize] >= 0 && !kept.contains(&expert) && !hits.contains(&expert)
        })
        .map(|expert| {
            let slot = slot_of[expert as usize] as usize;
            (state.layers[layer].stamps.load(slot), expert)
        })
        .collect::<Vec<_>>();
    // Protected hits can already occupy the whole share. Defer optional
    // seeds rather than evicting another layer's prompt working set.
    let protected = resident - others.len();
    seeded.truncate(share.saturating_sub(protected));
    let excess = (resident + seeded.len()).saturating_sub(share);
    if excess > 0 {
        others.sort_unstable();
        for (_, expert) in others.into_iter().take(excess) {
            let discarded = state
                .directory
                .discard(layer, expert)
                .expect("a streamed layer of the directory");
            if let Some(location) = discarded {
                unmap(state, source, location);
            }
        }
    }
    seeded
}

/// A scratch-served launch's fill (spec 4.9): of the routed experts the
/// scratch banks lack, list those in the layer's bank for the copy, then
/// stage the rest round by round, each
/// source reads bounded by idle intervals and device arrivals by `PARK_BOUND`.
/// Round `r`'s records
/// are freed once the device arrives at round `r + 1`, after converting
/// them; the last rounds' once the device is past the layer.
fn serve_scratch(state: &mut State, source: &ExpertSource, layer: usize, sequence: i32) {
    state.layers[layer].scratch_served = sequence;
    state.prompt = state.prompt.max(sequence);
    let point = state.clock(sequence, layer);
    advance(state, point, source);
    let held = match state.scratch.holds.take() {
        Some((held_layer, held)) if held_layer == layer => held,
        _ => vec![0; state.scratch.experts],
    };
    let flags = (0..state.scratch.experts)
        .map(|expert| state.scratch.flags.load(expert))
        .collect::<Vec<_>>();
    let flagged = (0..state.scratch.experts)
        .filter(|&expert| flags[expert] != 0 && held[expert] == 0)
        .map(|expert| expert as i32)
        .collect::<Vec<_>>();
    // The launch's experts resolve against the shared directory as a decode
    // request does: its hits are pinned while the fill copies them into the
    // scratch banks. The fill reads and converts every absent expert into the
    // scratch banks. At the layer's last scratch member the bank keeps the
    // scratch members' most routed experts, up to its share of the bank group:
    // the fill seeds them from the scratch banks into their slots and the
    // device commits them (`routed_seed_commit`). The decode that follows
    // reads what the prompt read most. A prompt reads every layer in turn:
    // past its share, a layer's seeds would evict the other layers' before
    // the next prompt reads them, and the bank would keep none of it.
    let (hits, absent): (Vec<i32>, Vec<i32>) = flagged
        .iter()
        .partition(|&&expert| state.layers[layer].slot_of[expert as usize] >= 0);
    if let Some(steps) = &mut state.trace {
        let trace = steps.entry(sequence).or_default();
        trace.scratch_requests += 1;
        trace.scratch_held += held.iter().filter(|&&count| count > 0).count();
        trace.scratch_misses += absent.len();
    }
    let routed = held
        .iter()
        .zip(&flags)
        .map(|(&held, &flag)| held + flag.max(0) as u32)
        .collect::<Vec<_>>();
    let seeded = if state.seed_steps.contains(&(sequence, layer)) {
        keep_routed(state, source, layer, sequence, &routed, &hits)
    } else {
        Vec::new()
    };
    let request = hits.iter().chain(&seeded).copied().collect::<Vec<_>>();
    let mut copies = Vec::new();
    let mut seeds = Vec::new();
    let mut cold = absent;
    if !request.is_empty() {
        let clock = point;
        state
            .directory
            .sync_stamps(
                layer,
                (0..state.layers[layer].stamps.len()).map(|s| state.layers[layer].stamps.load(s)),
            )
            .expect("shared GPU stamps");
        let resolution = state
            .directory
            .begin(clock, &request)
            .expect("a scratch request within its bank group");
        for victim in &resolution.evicted {
            unmap(state, source, *victim);
        }
        for (&expert, choice) in request.iter().zip(&resolution.choices) {
            let slot = choice.location.slot;
            if !choice.needs_conversion {
                copies.push((slot as u32, expert as usize));
                continue;
            }
            state.layers[layer].commits.store(slot, 0);
            state.layers[layer].slot_of[expert as usize] = slot as i32;
            state.layers[layer].expert_of[slot] = expert;
            state.layers[layer].stamps.store(slot, sequence);
            for tensor in state.layers[layer].stored {
                source.device_residency(tensor, expert as usize, true);
            }
            seeds.push((expert as u32, slot));
        }
        state
            .directory
            .submitted(clock)
            .expect("immutable scratch publication");
    }
    let failure = |state: &mut State, error: String| {
        state
            .fetch_failures
            .entry(sequence)
            .or_insert_with(|| format!("layer {layer}: source list transfer: {error}"));
        state.layers[layer].scratch_control.store(FAILURE, sequence);
    };
    if let Err(error) = publish_list(&state.scratch.seeds, sequence, &seeds) {
        failure(state, error);
        return;
    }
    for list in &state.scratch.copies {
        if let Err(error) = publish_list(list, sequence, &copies) {
            failure(state, error);
            return;
        }
    }
    let pools = state.layers[layer].stored.map(|stored| stored.pool);
    convey(state, source, layer, sequence);
    // Experts the conveyor already read go first, so the device converts
    // them while the rest arrive.
    cold.sort_by_key(|&expert| {
        state
            .staged
            .get(&(layer, expert))
            .and_then(|staged| staged.ticket.poll())
            != Some(Outcome::Done)
    });
    let chunks = cold
        .chunks(state.scratch.round_records.max(1))
        .map(<[i32]>::to_vec)
        .collect::<Vec<_>>();
    // Records of published rounds the device may still be converting, with
    // the expert each holds. Unpublished failed rounds stay ticket-owned.
    let mut held: Vec<Vec<(i32, [u32; KINDS])>> = Vec::new();
    let mut failed = chunks.len() > state.scratch.rounds();
    if failed {
        state.fetch_failures.entry(sequence).or_insert_with(|| {
            format!(
                "layer {layer}: scratch fill needs {} rounds, plan holds {}",
                chunks.len(),
                state.scratch.rounds()
            )
        });
    }
    for round in 0..state.scratch.rounds() {
        let deadline = Instant::now() + PARK_BOUND;
        if failed {
            break;
        }
        let experts = chunks.get(round).cloned().unwrap_or_default();
        // Round `round - 2`'s records are free once the device arrives at
        // round `round - 1`, past their convert.
        if round >= 2 && !experts.is_empty() {
            let arrival = state.scratch.arrival(round - 1);
            while state.layers[layer].scratch_control.load(arrival) != sequence {
                if Instant::now() >= deadline {
                    let observed = state.layers[layer].scratch_control.load(arrival);
                    state.fetch_failures.entry(sequence).or_insert_with(|| format!(
                        "layer {layer}, scratch round {round}: device arrival {} stayed at {observed}, expected {sequence}",
                        round - 1));
                    failed = true;
                    break;
                }
                state.heartbeat.beat();
                std::hint::spin_loop();
            }
            if failed {
                break;
            }
            for (expert, records) in std::mem::take(&mut held[round - 2]) {
                free_records(state, layer, records, (expert >= 0).then_some(expert));
            }
            convey(state, source, layer, sequence);
        }
        let mut records = Vec::with_capacity(experts.len());
        let mut ranges = Vec::with_capacity(experts.len() * KINDS);
        let mut fetched = Vec::new();
        let mut round_tickets = HashMap::new();
        for &expert in &experts {
            if let Some(kept) = revive(state, layer, expert) {
                records.push((expert, kept));
                continue;
            }
            // The conveyor's read of the expert becomes this round's.
            if let Some(staged) = state.staged.remove(&(layer, expert)) {
                if staged.ticket.priority() == Priority::Prefetch {
                    source.promote(&staged.ticket);
                }
                records.push((expert, staged.records));
                round_tickets.insert(expert, staged.ticket);
                continue;
            }
            if !pools.iter().all(|&pool| available(state, pool, Shelf::Own)) {
                reclaim_prefetches(state, source, layer);
            }
            if !pools.iter().all(|&pool| available(state, pool, Shelf::Own)) {
                let free = pools.map(|pool| state.pools[pool].free[Shelf::Own as usize].len());
                state.fetch_failures.entry(sequence).or_insert_with(|| {
                    format!(
                    "layer {layer}, scratch round {round}: staging exhausted (own free {free:?})")
                });
                failed = true;
                break;
            }
            let mut record = [0u32; KINDS];
            let stored = state.layers[layer].stored;
            for (kind, tensor) in stored.iter().enumerate() {
                record[kind] = take_record(state, tensor.pool, Shelf::Own).expect("checked free");
                ranges.push(Range {
                    file: tensor.file,
                    offset: tensor.base
                        + u64::try_from(expert).expect("an expert index") * tensor.bytes,
                    bytes: usize::try_from(tensor.bytes).expect("a record fits the host"),
                    destination: state.pools[tensor.pool].at(record[kind], tensor.bytes),
                });
            }
            records.push((expert, record));
            fetched.push(expert);
        }
        if !ranges.is_empty() {
            if let Some(steps) = &mut state.trace {
                steps.entry(sequence).or_default().source_bytes[0] +=
                    ranges.iter().map(|range| range.bytes as u64).sum::<u64>();
            }
            let ticket = source.submit_bulk(ranges, Priority::Demand);
            for expert in fetched {
                round_tickets.insert(expert, Arc::clone(&ticket));
            }
        }
        for ticket in round_tickets.values() {
            let began = state.trace.as_ref().map(|_| Instant::now());
            let outcome = wait_ticket(ticket, &mut state.heartbeat);
            let elapsed = began.map(|began| began.elapsed());
            if let Some(elapsed) = elapsed {
                state
                    .trace
                    .as_mut()
                    .expect("enabled trace")
                    .entry(sequence)
                    .or_default()
                    .scratch_ticket_wait += elapsed;
            }
            if outcome != Some(Outcome::Done) {
                failed = true;
                state.fetch_failures.entry(sequence).or_insert_with(|| format!(
                    "layer {layer}, scratch round {round}: fetch made no progress for {PARK_BOUND:?}: {}",
                    ticket.diagnostic()));
            } else {
                state.record_source(sequence, SourcePath::Scratch, ticket);
            }
        }
        if failed {
            // This round was never published to the device. A timed-out read
            // still owns its destination: release_past must observe its ticket
            // finish before the staging record can be reused.
            for (expert, records) in records {
                let ticket = round_tickets
                    .remove(&expert)
                    .unwrap_or_else(|| Ticket::done(Priority::Demand));
                state.staged.insert(
                    (layer, expert),
                    Staged {
                        records,
                        ticket,
                        sequence,
                    },
                );
            }
            break;
        }
        let device_staging = state.device_staging();
        if device_staging {
            for (index, &(_, record)) in records.iter().enumerate() {
                let destination = (round % 2) * state.scratch.round_records + index;
                for kind in 0..KINDS {
                    state.heartbeat.beat();
                    if let Err(error) = state.upload_source(
                        layer,
                        sequence,
                        SourcePath::Scratch,
                        kind,
                        record[kind],
                        destination,
                    ) {
                        state.fetch_failures.entry(sequence).or_insert_with(|| {
                            format!(
                            "layer {layer}, scratch round {round}: device source transfer: {error}")
                        });
                        failed = true;
                        break;
                    }
                }
                if failed {
                    break;
                }
            }
            if failed {
                // All source reads have completed. The unpublished copies
                // have no device consumer; retain host records until the
                // failed step's physical completion like published rounds.
                held.push(records);
                break;
            }
        }
        for (kind, list) in state.scratch.lists[round].iter().enumerate() {
            let entries = records
                .iter()
                .enumerate()
                .map(|(index, &(expert, record))| {
                    let source = if device_staging {
                        ((round % 2) * state.scratch.round_records + index) as u32
                    } else {
                        record[kind]
                    };
                    (source, expert as usize)
                })
                .collect::<Vec<_>>();
            if let Err(error) = publish_list(list, sequence, &entries) {
                failure(state, error);
                failed = true;
                break;
            }
        }
        held.push(records);
        if failed {
            break;
        }
        state.layers[layer]
            .scratch_control
            .store(state.scratch.acknowledgement(round), sequence);
    }
    if failed {
        state.layers[layer].scratch_control.store(FAILURE, sequence);
    } else {
        state.scratch.holds = Some((layer, routed));
    }
    // The device may still read these until it is past the layer.
    for (expert, records) in held.into_iter().flatten() {
        state.converted.push((sequence, layer, records, expert));
    }
    convey(state, source, layer, sequence);
}

/// The conveyor (spec 4.9): the next streamed layer's experts the bank lacks
/// are read while the device runs this one, behind every demand read; that
/// layer's fill takes them over. A fill tops it up as its rounds free
/// staging, so the source stays busy while the device converts. A read is
/// kept until the group's last step is past that layer. It reads into bank
/// records the prompt borrows, when a bank group lends them.
fn convey(state: &mut State, source: &ExpertSource, layer: usize, sequence: i32) {
    if layer + 1 < state.layers.len() && borrow(state, source) {
        let last = state.last_of_group(sequence);
        for expert in 0..state.scratch.experts as i32 {
            prefetch(state, source, layer + 1, expert, last, Window::Conveyor);
        }
    }
}

/// Write a convert or copy list for step `sequence`: its (source, target)
/// pairs, then the count, then the sequence, which the device checks against
/// its own step before it reads the rest (`routed_convert_wide`, `routed_copy`).
fn publish_list(list: &SharedWords, sequence: i32, entries: &[(u32, usize)]) -> Result<(), String> {
    for (index, &(source, target)) in entries.iter().enumerate() {
        list.store(2 + 2 * index, source as i32);
        list.store(3 + 2 * index, target as i32);
    }
    list.store(1, entries.len() as i32);
    list.store(0, sequence);
    list.publish()
}

/// Cancel every unstarted hint prefetch of the other layers so a demand read
/// of `layer` finds the pools' own records.
fn reclaim_prefetches(state: &mut State, source: &ExpertSource, layer: usize) {
    let keys = state
        .staged
        .iter()
        .filter(|(key, staged)| {
            let pool = &state.pools[state.layers[key.0].stored[0].pool];
            key.0 != layer
                && staged.ticket.priority() == Priority::Prefetch
                && pool.shelf(staged.records[0]) == Shelf::Own
        })
        .map(|(key, _)| *key)
        .collect::<Vec<_>>();
    for key in keys {
        // An arrived prefetch is a prediction's copy: given up like a queued
        // one. A started read runs to its end.
        let outcome = state.staged[&key].ticket.poll();
        if outcome.is_some() || source.cancel(&state.staged[&key].ticket) {
            let staged = state.staged.remove(&key).expect("present");
            let holds = (outcome == Some(Outcome::Done)).then_some(key.1);
            free_records(state, key.0, staged.records, holds);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{BackendName, DeviceCatalog, Element, Tensor};
    fn words(device: &seismic::Device, n: u64) -> SharedWords {
        SharedWords::new(Tensor::shared(device, Element::i32(), &[n]).unwrap()).unwrap()
    }
    #[test]
    fn progressing_source_batch_can_exceed_one_park_interval() {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(if cfg!(target_os = "macos") {
                BackendName::Metal
            } else {
                BackendName::Cuda
            })
            .unwrap();
        let ticket = Ticket::controlled_ranges(5);
        let worker_ticket = Arc::clone(&ticket);
        let worker = std::thread::spawn(move || {
            for _ in 0..5 {
                std::thread::sleep(PARK_BOUND / 3);
                worker_ticket.complete_controlled_range();
            }
        });
        let started = Instant::now();
        let outcome = wait_ticket(&ticket, &mut Heartbeat::new(words(&device, 1)));
        worker.join().unwrap();
        assert_eq!(outcome, Some(Outcome::Done));
        assert!(started.elapsed() > PARK_BOUND);
    }

    #[test]
    fn first_record_publishes_before_last_source_and_neither_becomes_a_cache_hit() {
        publication_case(false);
        publication_case(true);
    }
    /// The conveyor reads only into bank records a prompt has borrowed, and
    /// gives them back, forgetting what they keep, once the device is past
    /// the prompt and none is taken.
    #[test]
    fn the_conveyor_borrows_bank_records_for_a_prompt_and_gives_them_back() {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(if cfg!(target_os = "macos") {
                BackendName::Metal
            } else {
                BackendName::Cuda
            })
            .unwrap();
        let layers = vec![StreamedLayer::new(
            words(&device, 256),
            words(&device, 8),
            words(&device, 8),
            words(&device, 68),
            SharedWords::new(Tensor::shared(&device, Element::i32(), &[5, 64]).unwrap()).unwrap(),
            words(&device, 18),
            [words(&device, 66), words(&device, 66)],
            std::array::from_fn(|pool| StoredTensor {
                file: 0,
                base: 0,
                bytes: 16,
                pool,
            }),
        )];
        let pools = (0..3)
            .map(|_| {
                StagingPool::new(
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[64, 2, 2]).unwrap())
                        .unwrap(),
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[16, 2, 2]).unwrap())
                        .unwrap(),
                    64,
                    16,
                    16,
                    16,
                )
            })
            .collect();
        let file =
            std::env::temp_dir().join(format!("expert-conveyor-loan-{}", std::process::id()));
        std::fs::write(&file, [0u8; 512]).unwrap();
        let source = ExpertSource::new(vec![
            crate::streaming::source::ModelFile::open(&file).unwrap(),
        ])
        .unwrap();
        let runtime = StreamingRuntime::new(
            layers,
            ScratchFill::new(
                words(&device, 256),
                vec![],
                [
                    words(&device, 514),
                    words(&device, 514),
                    words(&device, 514),
                ],
                words(&device, 514),
                256,
                40,
            ),
            Heartbeat::new(words(&device, 1)),
            pools,
            source,
            SharedDirectory::controlled(),
            Some(Loan::new(0, 6..8)),
        )
        .unwrap();
        let mut state = runtime.state.lock().unwrap();
        assert_eq!(
            take_record(&mut state, 0, Shelf::Conveyor),
            None,
            "nothing borrowed"
        );
        assert!(borrow(&mut state, &runtime.source));
        assert_eq!(state.directory.share(0), Ok(6));
        let records: [u32; KINDS] =
            std::array::from_fn(|pool| take_record(&mut state, pool, Shelf::Conveyor).unwrap());
        assert!(records.iter().all(|&record| (64..80).contains(&record)));
        state.prompt = 1;
        free_records(&mut state, 0, records, Some(5));
        assert!(state.kept.contains_key(&(0, 5)));
        // A decode fill converts from the pools' own records only: its read
        // of expert 5 never takes back the conveyor's.
        let source = Arc::clone(&runtime.source);
        assert!(fetch(
            &mut state,
            &source,
            0,
            5,
            Priority::Demand,
            Shelf::Own,
            1
        ));
        assert!(
            state.staged[&(0, 5)]
                .records
                .iter()
                .all(|&record| record < 64)
        );
        assert!(state.kept.contains_key(&(0, 5)));
        // The device is not yet past the prompt: the records stay lent.
        give_back(&mut state);
        assert!(state.loan.as_ref().unwrap().lent);
        state.progress = state.clock(1, usize::MAX);
        let taken: [u32; KINDS] =
            std::array::from_fn(|pool| take_record(&mut state, pool, Shelf::Conveyor).unwrap());
        assert!(
            state.kept.contains_key(&(0, 5)),
            "the longest free are taken first"
        );
        give_back(&mut state);
        assert!(
            state.loan.as_ref().unwrap().lent,
            "conveyor records are taken"
        );
        free_records(&mut state, 0, taken, None);
        give_back(&mut state);
        assert!(!state.loan.as_ref().unwrap().lent);
        assert_eq!(state.directory.share(0), Ok(8));
        assert!(
            !state.kept.contains_key(&(0, 5)),
            "the bank overwrites what they kept"
        );
        assert_eq!(take_record(&mut state, 0, Shelf::Conveyor), None);
        drop(state);
        drop(runtime);
        std::fs::remove_file(file).unwrap();
    }
    fn publication_case(fail_last: bool) {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(if cfg!(target_os = "macos") {
                BackendName::Metal
            } else {
                BackendName::Cuda
            })
            .unwrap();
        let control_tensor = Tensor::shared(&device, Element::i32(), &[68]).unwrap();
        let placement_tensor = Tensor::shared(&device, Element::i32(), &[5, 64]).unwrap();
        let control = SharedWords::new(control_tensor.clone()).unwrap();
        let placement = SharedWords::new(placement_tensor.clone()).unwrap();
        let layers = vec![StreamedLayer::new(
            words(&device, 256),
            words(&device, 8),
            words(&device, 8),
            SharedWords::new(control_tensor).unwrap(),
            SharedWords::new(placement_tensor).unwrap(),
            words(&device, 18),
            [words(&device, 66), words(&device, 66)],
            std::array::from_fn(|pool| StoredTensor {
                file: 0,
                base: 0,
                bytes: 16,
                pool,
            }),
        )];
        let scratch = ScratchFill::new(
            words(&device, 256),
            vec![],
            [
                words(&device, 514),
                words(&device, 514),
                words(&device, 514),
            ],
            words(&device, 514),
            256,
            40,
        );
        let pools = (0..3)
            .map(|_| {
                StagingPool::new(
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[64, 2, 2]).unwrap())
                        .unwrap(),
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[16, 2, 2]).unwrap())
                        .unwrap(),
                    64,
                    16,
                    16,
                    16,
                )
            })
            .collect();
        let file =
            std::env::temp_dir().join(format!("expert-ready-control-{}", std::process::id()));
        std::fs::write(&file, [0u8; 512]).unwrap();
        let source = ExpertSource::new(vec![
            crate::streaming::source::ModelFile::open(&file).unwrap(),
        ])
        .unwrap();
        let runtime = StreamingRuntime::new(
            layers,
            scratch,
            Heartbeat::new(words(&device, 1)),
            pools,
            source,
            SharedDirectory::controlled(),
            None,
        )
        .unwrap();
        let last = Ticket::controlled();
        {
            let mut state = runtime.state.lock().unwrap();
            for expert in [0, 1] {
                let records =
                    std::array::from_fn(|pool| take_record(&mut state, pool, Shelf::Own).unwrap());
                state.staged.insert(
                    (0, expert),
                    Staged {
                        records,
                        ticket: if expert == 0 {
                            Ticket::done(Priority::Demand)
                        } else {
                            Arc::clone(&last)
                        },
                        sequence: 1,
                    },
                );
            }
        }
        control.store(COUNT, 4);
        for (index, expert) in [0, 0, 1, 1].into_iter().enumerate() {
            control.store(EXPERTS + index, expert);
        }
        control.store(REQUEST, 1);
        runtime.steps_submitted(1, 1, [None]);
        let deadline = Instant::now() + Duration::from_secs(2);
        // Each choice publishes its own flag, including duplicate choices.
        while (placement.load(4 * 64) != 1 || placement.load(4 * 64 + 1) != 1)
            && Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        assert_eq!(placement.load(4 * 64), 1);
        assert_eq!(placement.load(4 * 64 + 1), 1);
        assert_eq!(placement.load(4 * 64 + 2), 0);
        assert_eq!(control.load(ACKNOWLEDGED), 0);
        assert_eq!(control.load(FAILURE), 0);
        last.finish_controlled(fail_last);
        let terminal = if fail_last { FAILURE } else { ACKNOWLEDGED };
        while control.load(terminal) != 1 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(control.load(terminal), 1);
        // Wait for host publication to finish (the same mutex serializes
        // completion). Source completion has produced no conversion commits.
        {
            let state = runtime.state.lock().unwrap();
            for c in 0..4 {
                let expected = if fail_last && c >= 2 { 0 } else { 1 };
                assert_eq!(state.layers[0].placement.load(4 * 64 + c), expected);
            }
            assert_eq!(state.layers[0].slot_map.load(0), -1);
            assert_eq!(state.layers[0].slot_map.load(1), -1);
        }
        assert!(runtime.step_completed(1, true).is_err());
        drop(runtime);
        std::fs::remove_file(file).unwrap();
    }

    /// A layer-major group's later steps at a layer find what the earlier
    /// steps' fills left in the scratch banks and read only the experts they
    /// lack; its last scratch step seeds even before a small decode tail.
    #[test]
    fn a_prompt_group_reads_what_scratch_lacks_and_keeps_its_most_routed_experts() {
        let device = DeviceCatalog::discover()
            .unwrap()
            .open_backend(if cfg!(target_os = "macos") {
                BackendName::Metal
            } else {
                BackendName::Cuda
            })
            .unwrap();
        // Two views of one device tensor: the runtime's and the device's.
        let shared = |n| {
            let tensor = Tensor::shared(&device, Element::i32(), &[n]).unwrap();
            (
                SharedWords::new(tensor.clone()).unwrap(),
                SharedWords::new(tensor).unwrap(),
            )
        };
        let (commits, device_commits) = shared(8);
        let (scratch_control, device_control) = shared(18);
        let layers = vec![StreamedLayer::new(
            words(&device, 256),
            words(&device, 8),
            commits,
            words(&device, 68),
            SharedWords::new(Tensor::shared(&device, Element::i32(), &[5, 64]).unwrap()).unwrap(),
            scratch_control,
            [words(&device, 66), words(&device, 66)],
            std::array::from_fn(|pool| StoredTensor {
                file: 0,
                base: 0,
                bytes: 16,
                pool,
            }),
        )];
        let (flags, device_flags) = shared(256);
        let lists = [shared(514), shared(514), shared(514)];
        let copies = [shared(514), shared(514), shared(514)];
        let (seeds, device_seeds) = shared(514);
        let [lists, device_lists] = {
            let [(a, x), (b, y), (c, z)] = lists;
            [[a, b, c], [x, y, z]]
        };
        let [copies, device_copies] = {
            let [(a, x), (b, y), (c, z)] = copies;
            [[a, b, c], [x, y, z]]
        };
        let scratch = ScratchFill::new(flags, vec![lists], copies, seeds, 256, 40);
        let pools = (0..3)
            .map(|_| {
                StagingPool::new(
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[64, 2, 2]).unwrap())
                        .unwrap(),
                    SharedBytes::new(Tensor::shared(&device, Element::f32(), &[16, 2, 2]).unwrap())
                        .unwrap(),
                    64,
                    16,
                    16,
                    16,
                )
            })
            .collect();
        let file = std::env::temp_dir().join(format!("expert-union-fill-{}", std::process::id()));
        std::fs::write(&file, [0u8; 512]).unwrap();
        let source = ExpertSource::new(vec![
            crate::streaming::source::ModelFile::open(&file).unwrap(),
        ])
        .unwrap();
        let runtime = StreamingRuntime::new(
            layers,
            scratch,
            Heartbeat::new(words(&device, 1)),
            pools,
            source,
            SharedDirectory::controlled(),
            None,
        )
        .unwrap();
        // Three scratch members followed by a decode member. The latter
        // has no scratch fill from which it could seed the bank.
        runtime.steps_submitted(1, 4, [Some(3)]);
        let listed = |list: &SharedWords, at: usize| {
            (0..list.load(1) as usize)
                .map(|i| list.load(at + 2 * i))
                .collect::<Vec<_>>()
        };
        let mut seeded = Vec::new();
        // (expert, flag): the choices that route to it.
        for (sequence, routed) in [
            (1, &[(3, 1), (4, 2)][..]),
            (2, &[(4, 1), (5, 1)][..]),
            (
                3,
                &[(6, 1), (7, 1), (8, 1), (9, 2), (10, 2), (11, 2), (12, 2)][..],
            ),
        ] {
            device_flags.fill(0);
            for &(expert, flag) in routed {
                device_flags.store(expert as usize, flag);
            }
            device_control.store(REQUEST, sequence);
            let deadline = Instant::now() + Duration::from_secs(2);
            while device_control.load(SCRATCH_HEADER) != sequence && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert_eq!(device_control.load(SCRATCH_HEADER), sequence);
            assert_eq!(device_control.load(FAILURE), 0);
            assert_eq!(device_lists[0].load(0), sequence);
            seeded.push(listed(&device_seeds, 2));
            let mut read = listed(&device_lists[0], 3);
            read.sort();
            let copied =
                (device_copies[0].load(0) == sequence).then(|| listed(&device_copies[0], 3));
            match sequence {
                1 => assert_eq!(read, [3, 4]),
                2 => assert_eq!(read, [5]),
                _ => assert_eq!(read, [6, 7, 8, 9, 10, 11, 12]),
            }
            assert_eq!(copied.unwrap_or_default(), Vec::<i32>::new());
            // The device commits the step's seeds before it runs on.
            for i in 0..device_seeds.load(1) as usize {
                device_commits.store(device_seeds.load(3 + 2 * i) as usize, sequence);
            }
        }
        // Only the group's last scratch step seeds: the layer's share (all 8
        // records) of the group's most routed experts, the group's choices
        // summed over its steps. Experts 7 and 8 are read but not kept.
        assert_eq!(seeded, [vec![], vec![], vec![4, 9, 10, 11, 12, 3, 5, 6]]);
        runtime.step_completed(1, true).unwrap();
        runtime.step_completed(2, true).unwrap();
        runtime.step_completed(3, true).unwrap();
        runtime.step_completed(4, true).unwrap();
        // A later prompt needs every cached expert and a more popular cold
        // one. All hits must remain pinned for copying; optional seeding may
        // not reserve a ninth slot in this eight-slot bank.
        runtime.state.lock().unwrap().scratch.holds = None;
        runtime.steps_submitted(5, 1, [Some(5)]);
        device_flags.fill(0);
        for &expert in &seeded[2] {
            device_flags.store(expert as usize, 1);
        }
        device_flags.store(13, 10);
        device_control.store(REQUEST, 5);
        let deadline = Instant::now() + Duration::from_secs(2);
        while device_control.load(SCRATCH_HEADER) != 5 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert_eq!(device_control.load(SCRATCH_HEADER), 5);
        assert_eq!(device_control.load(FAILURE), 0);
        assert_eq!(listed(&device_lists[0], 3), [13]);
        assert_eq!(listed(&device_seeds, 2), Vec::<i32>::new());
        let mut copied = listed(&device_copies[0], 3);
        copied.sort();
        let mut cached = seeded[2].clone();
        cached.sort();
        assert_eq!(copied, cached);
        runtime.step_completed(5, true).unwrap();

        // A failed scratch round must keep a still-running source's staging
        // even after device progress passes the failed layer.
        runtime.steps_submitted(6, 1, [Some(6)]);
        let delayed = Ticket::controlled();
        {
            let mut state = runtime.state.lock().unwrap();
            state.scratch.holds = None;
            device_flags.fill(0);
            device_flags.store(14, 1);
            let records =
                std::array::from_fn(|pool| take_record(&mut state, pool, Shelf::Own).unwrap());
            state.staged.insert(
                (0, 14),
                Staged {
                    records,
                    ticket: Arc::clone(&delayed),
                    sequence: 6,
                },
            );
            // Keep the emulated device request consistent with the manual
            // serve, so the watcher cannot rediscover the older request 5.
            device_control.store(REQUEST, 6);
            serve_scratch(&mut state, &runtime.source, 0, 6);
            assert_eq!(device_control.load(FAILURE), 6);
            assert!(state.fetch_failures[&6].contains("running=1"));
            state.progress = state.clock(6, usize::MAX);
            release_past(&mut state, &runtime.source);
            assert!(state.staged.contains_key(&(0, 14)));
            for (pool, record) in records.into_iter().enumerate() {
                assert!(!state.pools[pool].free[Shelf::Own as usize].contains(&record));
            }
            delayed.finish_controlled(true);
            release_past(&mut state, &runtime.source);
            assert!(!state.staged.contains_key(&(0, 14)));
            assert!(!state.kept.contains_key(&(0, 14)));
            for (pool, record) in records.into_iter().enumerate() {
                assert!(state.pools[pool].free[Shelf::Own as usize].contains(&record));
            }
        }
        assert!(runtime.step_completed(6, true).is_err());
        drop(runtime);
        std::fs::remove_file(file).unwrap();
    }
}
