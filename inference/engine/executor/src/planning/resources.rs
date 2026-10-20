use super::{ModelLoadPlan, PlannedMethod, source_import_peak_bytes, weight_bytes_by_component};
use crate::{
    PreparedDrafterGraphs, PreparedStateCopyGraphs, PreparedTargetGraphs,
    PreparedTargetReadoutGraphs, PreparedVisionGraphs,
};
use magnitude_family_contracts::ModelDefinition;
use magnitude_state::{
    BankCapacity, ComponentDescriptor, ComponentSpec, HistoryDomainId, HistoryDomainKind,
    HistoryDomainLayout, HistoryDomainPlan, HistoryDomainTrace, KvCodec, LayerRef,
    ModelStateLayout, StateStore, StoreBindings,
};
use seismic::{BackendName, Device, Element, SlabLayout, SlabRegion};
use std::num::NonZeroUsize;
use std::rc::Rc;

/// The service's bounds a load plans for. None is a request-count batch
/// width: a launch is bounded by its token budget (rows and request slots
/// are compiled shape classes), its vocabulary work by the selection bound,
/// and every per-request resource beyond what one request needs grows
/// elastically under heap claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Rows one launch admits: the larger step token budget.
    pub max_launch_rows: usize,
    /// Requests one launch serves: its largest request-slot class.
    pub max_launch_slots: usize,
    /// Target rows one launch selects a token for. The selection bound: every
    /// vocabulary-wide buffer of the target readout is sized by it.
    pub max_selected_rows: usize,
    /// Requests one drafter launch drafts for: the selection bound of the
    /// head and separate draft.
    pub max_drafting_slots: usize,
    /// Rows one launch may export full logits for. Zero for a served load,
    /// which prepares no logits class; diagnostics that read logits set it.
    pub exported_logits_rows: usize,
    pub max_images_per_request: usize,
    pub max_image_cells: usize,
    /// Queue each continuable target step's successor before the step
    /// completes (cross-step pipelining): one more target launch in flight
    /// and one more successor bank per live request.
    pub lookahead: bool,
    /// The most chunks of one prompt a launch chain runs as a layer-major
    /// group, resolved from the requested `PrefillGroup` and whether the load
    /// streams. One is chunk-major.
    pub group_capacity: NonZeroUsize,
}

/// The prompt group a load requests: how many of a prompt's chunks one
/// launch chain may run together. Chunk-major runs each chunk through every
/// block before the next; a group of chained chunks runs block by block, each
/// block over every chunk of the group (layer-major), so a streamed load reads
/// each routed layer's experts once per group rather than once per chunk.
/// Every size computes the same values; a group holds all its chunks in flight
/// at once, and its chunks run on the device as one unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillGroup {
    /// `STREAMED_PREFILL_GROUP` chunks for a load that streams its routed
    /// experts, one for a resident load, which has no read to save.
    Auto,
    Chunks(NonZeroUsize),
}

/// Chunks of a streamed load's prompt group under `PrefillGroup::Auto`.
pub const STREAMED_PREFILL_GROUP: NonZeroUsize = NonZeroUsize::new(8).expect("nonzero");

impl PrefillGroup {
    /// The group capacity of a load that does or does not stream its routed
    /// experts.
    pub fn capacity(self, streams: bool) -> NonZeroUsize {
        match self {
            Self::Auto if streams => STREAMED_PREFILL_GROUP,
            Self::Auto => NonZeroUsize::MIN,
            Self::Chunks(chunks) => chunks,
        }
    }
}

impl std::str::FromStr for PrefillGroup {
    type Err = String;

    /// `auto`, or a chunk count (1 is chunk-major).
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            _ => value
                .parse::<NonZeroUsize>()
                .map(Self::Chunks)
                .map_err(|_| format!("expected auto or a chunk count of at least 1, got {value}")),
        }
    }
}

/// The most merged cells one image encodes, whatever its model declares:
/// about 4.2 MP at Qwen3-VL's 32-pixel cells. A larger image is resized
/// down to it.
pub const MAX_IMAGE_CELLS: usize = 4096;

/// The most merged cells one image of `definition` encodes in a load whose
/// launches carry `launch_rows` rows: the declared resize's bound within
/// [`MAX_IMAGE_CELLS`]. Prefill places an image's features across launches,
/// unless its rows attend each other in the decoder: such an image is one
/// launch. Zero without vision.
pub fn image_cell_limit(
    definition: &ModelDefinition,
    launch_rows: usize,
) -> Result<usize, crate::PlanError> {
    let Some(vision) = &definition.vision else {
        return Ok(0);
    };
    let declared = vision
        .max_cells()
        .map_err(|error| crate::PlanError::InvalidDefinition(error.to_string()))?;
    let declared = usize::try_from(declared).unwrap_or(usize::MAX);
    let bidirectional = definition.decoder.sublayers().any(|(_, sublayer)| {
        matches!(
            &sublayer.op,
            magnitude_family_contracts::Operator::Attention(attention)
                if attention.media_rows == magnitude_family_contracts::MediaRowAttention::Bidirectional
        )
    });
    let limit = declared.min(MAX_IMAGE_CELLS);
    let limit = if bidirectional {
        limit.min(launch_rows)
    } else {
        limit
    };
    if limit == 0 {
        return Err(crate::PlanError::InvalidDefinition(
            "the model's images encode no cells".into(),
        ));
    }
    Ok(limit)
}

impl ResourceLimits {
    /// Target launches in flight at once, the longest chain the device
    /// queues: a step and, with lookahead, its queued successor, or a prompt
    /// group's chunks.
    pub fn longest_chain(&self) -> usize {
        (1 + usize::from(self.lookahead)).max(self.group_capacity.get())
    }

    /// The graph slots a load commits at startup: what one request needs to
    /// run. Activations cover the launches in flight at once, whatever the
    /// request count; each holds its own upload regions, and every one binds
    /// the device's one workspace arena. The target's outputs are its
    /// residual pair: each block graph reads one and writes the other, and
    /// the device's submission order lets every launch reuse the same pair.
    /// A prompt group interleaves its chunks' blocks, so each chunk holds its
    /// own pair.
    /// Outputs a request retains after its launch completes (readout
    /// features, head drafts, encoded images) start at one request's need
    /// and grow one slot at a time under a heap claim.
    pub fn startup_slots(&self) -> StartupSlots {
        let launches = self.longest_chain();
        StartupSlots {
            target: GraphSlots {
                activations: launches,
                output: 2 * self.group_capacity.get(),
            },
            readout: GraphSlots {
                activations: launches,
                output: 1 + launches,
            },
            // A prompt chunk's drafter entry rides with each target launch
            // in flight.
            head: GraphSlots {
                activations: launches,
                output: 1 + launches,
            },
            // Vision holds nothing until an image arrives: a text-only
            // session never uses it. An encode claims storage of its image's
            // class (`VisionGraphPool`), outside the shared arena.
            vision: GraphSlots {
                activations: 0,
                output: 0,
            },
            state: GraphSlots {
                activations: 1,
                output: 1,
            },
        }
    }
}

/// One graph family's slot counts: concurrent activations, each with its
/// upload regions, and output arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphSlots {
    pub activations: usize,
    pub output: usize,
}

/// Every graph family's startup slots (see [`ResourceLimits::startup_slots`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartupSlots {
    pub target: GraphSlots,
    pub readout: GraphSlots,
    pub head: GraphSlots,
    pub vision: GraphSlots,
    pub state: GraphSlots,
}

/// Whether a device of `backend` has a form of the affine K8/V4 prefill
/// entry that reads history decoded for the call, for heads of `head_width`
/// columns: on Metal the direct form on tensor operations (any head) and
/// the co-issue form on simdgroup matrices (128-, 256- and 512-column heads,
/// a 512-column head as two 256-column windows); on CUDA the request-local
/// F16 operand-staging form for the affine entry's32..512-column head bands.
/// Such forms take the history row tiles their launch's rows see; every
/// other form reads the history in place.
pub fn reads_decoded_history(
    backend: BackendName,
    tensor_operations: bool,
    head_width: u64,
) -> bool {
    DecodedHistoryForm::of(backend, tensor_operations, head_width).is_some()
}

/// How a form of the affine K8/V4 prefill entry that reads history decoded
/// for the call takes the history row tiles its launch lists. The form
/// declares what its work follows, and with it the smallest listing class
/// worth sealing a graph for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodedHistoryForm {
    /// Metal's direct and co-issue forms decode the listed tiles into a
    /// window of the call's partial-output scratch, a round per window. A
    /// listing that fits the window is one round whatever its class, and a
    /// tile the launch does not see reads no rows, so classes below the
    /// floor would change nothing of what a launch does.
    Windowed,
    /// CUDA's operand-staging form stages the listed tiles into scratch
    /// sized by the class (`history_keys`, `history_values`). An entry of
    /// the list that names no tile stages nothing and is never looked up,
    /// so the form's work follows the tiles a launch sees whatever its
    /// class: one class of one request's worth serves every listing.
    Staged,
}

impl DecodedHistoryForm {
    /// The form a device of `backend` has for heads of `head_width`
    /// columns, if any ([`reads_decoded_history`]).
    pub fn of(backend: BackendName, tensor_operations: bool, head_width: u64) -> Option<Self> {
        match backend {
            BackendName::Metal
                if tensor_operations || matches!(head_width, 128 | 256 | 512) =>
            {
                Some(Self::Windowed)
            }
            BackendName::Cuda if matches!(head_width, 32 | 64 | 128 | 256 | 512) => {
                Some(Self::Staged)
            }
            _ => None,
        }
    }

    /// The tile count of the form's smallest listing class below one
    /// request's worth, if it has classes there.
    pub fn smallest_listing_tiles(self) -> Option<u64> {
        match self {
            Self::Windowed => Some(16),
            Self::Staged => None,
        }
    }

    /// Whether the form's listing classes are sealed for every history
    /// segment count, as the classes that list none are. Metal's forms
    /// reserve threadgroup memory per segment, so theirs are. CUDA's reads
    /// its rows' spans from a table of a few bytes a segment beside the
    /// operand stages, and does nothing for a span that is empty: its
    /// listing class is sealed for the most segments alone, and a launch
    /// of fewer pads each row's spans with empty ones.
    pub fn lists_every_segment_count(self) -> bool {
        match self {
            Self::Windowed => true,
            Self::Staged => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceCapacity {
    /// Stable capacity of the selected device's physical allocation domain.
    pub domain_bytes: u64,
    /// Whether the selected device forms Metal tensor operations: its own
    /// probe (`seismic::DeviceInfo::forms_tensor_operations`), which decides
    /// the graph classes a plan has, so assessment and load plan the same
    /// graphs. A plan for no concrete device states which it assumes.
    pub tensor_operations: TensorOperations,
}

/// Whether a device forms Metal tensor operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOperations {
    Formed,
    Absent,
}

impl TensorOperations {
    pub fn of(forms: bool) -> Self {
        if forms {
            Self::Formed
        } else {
            Self::Absent
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCapacityPlan {
    /// Largest number of one-row requests the physical domain could hold.
    pub live_requests: usize,
    /// Explicit branch checkpoints resident at once.
    pub checkpoints: usize,
    /// Cross-request prefix entries retention may hold.
    pub retention_entries: usize,
    pub history_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBytes {
    pub target_weights: u64,
    pub head_weights: u64,
    pub vision_weights: u64,
    pub history: u64,
    pub recurrent_banks: u64,
    /// Fixed native argument/result buffers owned by every attested entry.
    pub prepared_programs: u64,
    pub scratch: u64,
}

/// A Seismic-derived physical family charge at startup. The engine chooses
/// the slot counts; Seismic supplies every byte and every tensor layout
/// within each slot. Each activation holds the byte capacities in
/// `upload_regions`, allocated with it: one per graph run its lease keeps
/// in flight at once. Output slots beyond `output_slots` are elastic
/// run-time claims of `output_bytes` each. `workspace_bytes` is not part of
/// `committed_bytes`: every family binds the device's one workspace arena,
/// which the plan charges once at the largest family's workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeGraphCharge {
    pub workspace_bytes: u64,
    pub output_bytes: u64,
    pub upload_bytes: u64,
    pub upload_regions: Vec<u64>,
    pub activations: usize,
    pub output_slots: usize,
    pub committed_bytes: u64,
}

/// A family's per-slot storage: scratch, output arena and upload region
/// bytes, and the runs one workspace lease keeps in flight.
#[derive(Clone, Copy, Debug)]
struct FamilyFootprint {
    workspace_bytes: u64,
    output_bytes: u64,
    upload_bytes: u64,
    runs_in_flight: usize,
}

impl FamilyFootprint {
    /// A family whose lease runs one graph at a time to completion.
    fn serial(family: &seismic::NativeGraphFamily) -> Self {
        Self {
            workspace_bytes: family.workspace_bytes(),
            output_bytes: family.output_bytes(),
            upload_bytes: family.upload_bytes(),
            runs_in_flight: 1,
        }
    }
}

impl NativeGraphCharge {
    pub(crate) fn from_checked(
        storage: seismic::NativeGraphStorageBytes,
        runs_in_flight: usize,
        slots: GraphSlots,
    ) -> Result<Self, String> {
        Self::from_footprint(
            FamilyFootprint {
                workspace_bytes: storage.workspace,
                output_bytes: storage.output,
                upload_bytes: storage.upload,
                runs_in_flight,
            },
            slots,
        )
    }

    fn from_prepared(graphs: &PreparedTargetGraphs, slots: GraphSlots) -> Result<Self, String> {
        Self::from_footprint(
            FamilyFootprint {
                workspace_bytes: graphs.workspace_bytes(),
                output_bytes: graphs.output_bytes(),
                upload_bytes: graphs.family().upload_bytes(),
                runs_in_flight: graphs.runs_per_step(),
            },
            slots,
        )?.with_upload_regions(graphs.upload_regions())
    }

    /// A region for each stage's largest legal upload, rather than the
    /// family's largest upload repeated for every decoder block.
    pub(crate) fn with_upload_regions(mut self, regions: Vec<u64>) -> Result<Self, String> {
        if regions.len() != self.upload_regions.len()
            || regions.iter().any(|&bytes| bytes > self.upload_bytes)
        {
            return Err("upload regions differ from the family's admitted runs".into());
        }
        let per_activation = regions
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
            .ok_or("upload region sum overflows")?;
        self.committed_bytes = per_activation
            .checked_mul(self.activations as u64)
            .and_then(|bytes| {
                self.output_bytes
                    .checked_mul(self.output_slots as u64)
                    .and_then(|outputs| bytes.checked_add(outputs))
            })
            .ok_or("graph charge overflows")?;
        self.upload_regions = regions;
        Ok(self)
    }

    pub(crate) fn shared_upload_bytes(&self) -> Result<u64, String> {
        self.upload_regions.iter().try_fold(0u64, |total, &bytes| total.checked_add(bytes))
            .and_then(|bytes| bytes.checked_mul(self.activations as u64))
            .ok_or_else(|| "graph upload charge overflows".to_owned())
    }

    fn from_footprint(footprint: FamilyFootprint, slots: GraphSlots) -> Result<Self, String> {
        let GraphSlots {
            activations,
            output: output_slots,
        } = slots;
        let count = |value: usize| u64::try_from(value).map_err(|_| "slot count exceeds u64");
        let per_activation = footprint
            .upload_bytes
            .checked_mul(count(footprint.runs_in_flight)?)
            .ok_or("graph activation byte count overflows")?;
        let committed_bytes = per_activation
            .checked_mul(count(activations)?)
            .and_then(|bytes| {
                footprint
                    .output_bytes
                    .checked_mul(u64::try_from(output_slots).ok()?)
                    .and_then(|outputs| bytes.checked_add(outputs))
            })
            .ok_or("graph charge overflows")?;
        Ok(Self {
            workspace_bytes: footprint.workspace_bytes,
            output_bytes: footprint.output_bytes,
            upload_bytes: footprint.upload_bytes,
            upload_regions: vec![footprint.upload_bytes; footprint.runs_in_flight],
            activations,
            output_slots,
            committed_bytes,
        })
    }
}

/// The device's one workspace arena: the largest workspace of the `charges`
/// that hold activations from startup. A family that holds none (vision)
/// claims its own workspace with its first activation, so an image's
/// workspace never enlarges what every load holds.
pub(crate) fn arena_bytes(charges: &[&NativeGraphCharge]) -> u64 {
    charges
        .iter()
        .filter(|charge| charge.activations != 0)
        .map(|charge| charge.workspace_bytes)
        .max()
        .unwrap_or(0)
}

pub(crate) fn graph_shared_upload_bytes(charges: &[&NativeGraphCharge]) -> Result<u64, String> {
    charges.iter().try_fold(0u64, |total, charge| {
        total
            .checked_add(charge.shared_upload_bytes()?)
            .ok_or_else(|| "graph upload total overflows".to_owned())
    })
}

/// Every graph byte a load commits: the workspace arena, and each family's
/// activations and outputs.
pub(crate) fn graph_scratch_bytes(charges: &[&NativeGraphCharge]) -> Result<u64, String> {
    charges
        .iter()
        .try_fold(arena_bytes(charges), |bytes, charge| {
            bytes.checked_add(charge.committed_bytes)
        })
        .ok_or_else(|| "graph scratch byte count overflows".into())
}

/// Exact projection used to construct one numerical state arena. The planner
/// owns every capacity and byte fact; construction only materializes it.
/// `history` is the store's history domains as `StateStore::new` takes them;
/// `domains` projects each stored domain, indexed by `HistoryDomainId`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateStorePlan {
    pub context_rows: usize,
    /// The largest advance (slot rows): Window domains' row limit.
    pub max_advance: usize,
    pub history: Vec<HistoryDomainPlan>,
    pub domains: Vec<HistoryStorePlan>,
    pub recurrent_components: Vec<ComponentSpec>,
    pub bank_capacity: BankCapacity,
    pub recurrent_bank_bytes: u64,
    pub zero_seed_bytes: u64,
    pub recurrent_pool_bytes: u64,
}

/// One stored history domain of a store plan: exactly what the store derives
/// for it (see `StateStore::allocation_trace`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryStorePlan {
    pub kind: HistoryDomainKind,
    pub components: Vec<ComponentDescriptor>,
    /// Reserved row addresses, sealed into graphs.
    pub rows: usize,
    pub row_bytes: u64,
    pub slab_rows: u32,
    pub page_rows: u32,
    pub span_limit: usize,
}

/// The history an attention layer binds: its stored domain, and the
/// component whose regions it reads (its own, or a Shared domain's source).
#[derive(Clone, Copy, Debug)]
pub struct LayerHistory<'a> {
    pub domain: HistoryDomainId,
    pub store: &'a HistoryStorePlan,
    pub component: &'a ComponentDescriptor,
}

impl HistoryStorePlan {
    /// The most history row tiles (`SLAB_ROW_TILE` rows) the rows of one
    /// launch of `slots` requests see: each request's history lies in at most
    /// `span_limit` pages. The bound a launch's per-call storage of history
    /// rows is sized by, whatever the reservation `rows`.
    pub fn launch_tiles(&self, slots: u64) -> Result<u64, String> {
        let page_tiles = u64::from(self.page_rows) / magnitude_state::SLAB_ROW_TILE as u64;
        u64::try_from(self.span_limit)
            .ok()
            .and_then(|spans| spans.checked_mul(page_tiles))
            .and_then(|tiles| tiles.checked_mul(slots))
            .ok_or_else(|| "launch history tile count overflows".into())
    }

    /// The tile counts of the graph classes that list a launch's history
    /// row tiles for `form`, ascending: powers of two from the form's
    /// smallest class ([`DecodedHistoryForm::smallest_listing_tiles`]),
    /// then one request's worth (`launch_tiles(1)`). A launch takes the
    /// smallest that holds the tiles its rows see, so a form that attends
    /// the listed rows a part at a time dispatches at most twice the parts
    /// the launch needs.
    pub fn listed_tile_classes(&self, form: DecodedHistoryForm) -> Result<Vec<u64>, String> {
        let most = self.launch_tiles(1)?;
        let mut classes =
            std::iter::successors(form.smallest_listing_tiles(), |tiles| tiles.checked_mul(2))
                .take_while(|tiles| *tiles < most)
                .collect::<Vec<_>>();
        classes.push(most);
        Ok(classes)
    }

    fn trace(&self) -> HistoryDomainTrace {
        HistoryDomainTrace {
            kind: self.kind,
            capacity: self.rows,
            row_bytes: self.row_bytes,
            bytes: self.row_bytes * self.rows as u64,
            slab_rows: self.slab_rows as usize,
            page_rows: self.page_rows as usize,
            span_limit: self.span_limit,
        }
    }

    pub fn slab_layout(&self) -> Result<SlabLayout, String> {
        let regions = self
            .components
            .iter()
            .flat_map(ComponentDescriptor::planes)
            .map(|plane| SlabRegion {
                element: Element::dense(plane.dtype),
                row_shape: plane.row_extents.iter().map(|&size| size as u64).collect(),
            })
            .collect::<Vec<_>>();
        SlabLayout::for_regions(u64::from(self.slab_rows), self.rows as u64, &regions)
            .map_err(|error| error.to_string())
    }
}

impl StateStorePlan {
    /// The most spans any history of the store presents to one launch (a
    /// Shared reader included, see `magnitude_state::history_geometry`): the
    /// span class bound graphs are sealed to and launches are checked
    /// against (1 without history).
    pub fn span_limit(&self) -> usize {
        self.domains
            .iter()
            .map(|domain| domain.span_limit)
            .max()
            .unwrap_or(1)
    }

    /// The history `layer` binds, resolving a Shared layer to its source.
    /// `None` for a layer without history in this store.
    pub fn layer_history(&self, layer: LayerRef) -> Option<LayerHistory<'_>> {
        let owner = self
            .history
            .iter()
            .find_map(|plan| match &plan.layout {
                HistoryDomainLayout::Shared { source, layers } if layers.contains(&layer) => {
                    Some(*source)
                }
                _ => None,
            })
            .unwrap_or(layer);
        self.domains.iter().enumerate().find_map(|(index, store)| {
            store
                .components
                .iter()
                .find(|component| component.layer == owner)
                .map(|component| LayerHistory {
                    domain: HistoryDomainId(index),
                    store,
                    component,
                })
        })
    }

    /// The raw keys' (Window) and block keys' (Block) domains of target
    /// block `block`'s indexer (`LayerRef::Indexer`). `None` for a block
    /// without one.
    pub fn indexer_history(&self, block: u32) -> Option<(&HistoryStorePlan, &HistoryStorePlan)> {
        let layer = LayerRef::Indexer(block);
        let find = |blocks: bool| {
            self.domains.iter().find(|store| {
                matches!(store.kind, HistoryDomainKind::Block { .. }) == blocks
                    && store
                        .components
                        .iter()
                        .any(|component| component.layer == layer)
            })
        };
        Some((find(false)?, find(true)?))
    }

    /// The store's one history domain, for a store whose history the
    /// caller addresses as one domain (the draft head's).
    pub fn sole_history(&self) -> Result<&HistoryStorePlan, String> {
        match self.domains.as_slice() {
            [domain] => Ok(domain),
            domains => Err(format!(
                "store has {} history domains where one is required",
                domains.len()
            )),
        }
    }

    pub fn bank_slab_banks(&self) -> Result<u32, String> {
        u32::try_from(magnitude_state::banks_per_slab(self.recurrent_bank_bytes)?)
            .map_err(|_| "bank slab count exceeds u32".into())
    }

    pub fn bank_slab_layout(&self) -> Result<Option<SlabLayout>, String> {
        self.bank_slab_layout_for(self.recurrent_components.iter())
    }

    fn bank_slab_layout_for<'a>(
        &self,
        components: impl Iterator<Item = &'a ComponentSpec>,
    ) -> Result<Option<SlabLayout>, String> {
        let regions = components
            .map(|component| SlabRegion {
                element: Element::dense(component.dtype),
                row_shape: component.shape.iter().map(|&size| size as u64).collect(),
            })
            .collect::<Vec<_>>();
        if regions.is_empty() {
            return Ok(None);
        }
        let logical_banks = self
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?;
        SlabLayout::for_regions(
            u64::from(self.bank_slab_banks()?),
            logical_banks as u64,
            &regions,
        )
        .map(Some)
        .map_err(|error| error.to_string())
    }

    /// Every stored domain's address table and first slab.
    pub fn startup_history_bytes(&self) -> Result<u64, String> {
        self.domains.iter().try_fold(0u64, |total, domain| {
            let layout = domain.slab_layout()?;
            layout
                .address_table_bytes
                .checked_add(layout.slab_bytes)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| "initial history slab charge overflows".into())
        })
    }

    pub fn startup_bank_bytes(&self) -> Result<u64, String> {
        self.bank_slab_layout()?.map_or(Ok(0), |layout| {
            layout
                .address_table_bytes
                .checked_add(layout.slab_bytes)
                .ok_or_else(|| "initial bank slab charge overflows".into())
        })
    }

    /// One request's history at `depth` tokens: a Token domain holds `depth`
    /// rows, a Window domain its steady footprint of `n` plus one advance,
    /// independent of the depth; each rounded to whole slabs of its domain.
    pub fn history_bytes_at_depth(&self, depth: u64) -> Result<u64, String> {
        self.domains.iter().try_fold(0u64, |total, domain| {
            total
                .checked_add(Self::history_domain_bytes_at_depth(
                    domain,
                    self.max_advance,
                    depth,
                )?)
                .ok_or_else(|| "history slab fit charge overflows".into())
        })
    }

    fn history_domain_bytes_at_depth(
        domain: &HistoryStorePlan,
        max_advance: usize,
        depth: u64,
    ) -> Result<u64, String> {
        let layout = domain.slab_layout()?;
        let rows = match domain.kind {
            HistoryDomainKind::Token => depth,
            kind => u64::try_from(
                kind.row_limit(
                    usize::try_from(depth).map_err(|_| "fit depth exceeds host range")?,
                    max_advance,
                )
                .map_err(|error| error.to_string())?,
            )
            .map_err(|_| "window rows exceed u64")?,
        };
        // Whole pages: a history's first page may be partial once a
        // window releases the rows before it, its last page always may.
        let page = u64::from(domain.page_rows);
        let front = u64::from(matches!(domain.kind, HistoryDomainKind::Window { .. }));
        let pages = rows.max(1).div_ceil(page) + front;
        let slabs = (pages * page).div_ceil(u64::from(domain.slab_rows));
        layout
            .slab_bytes
            .checked_mul(slabs)
            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
            .ok_or_else(|| "history slab fit charge overflows".into())
    }

    /// The same slab recipes as ordinary fit, partitioned by physical owner.
    pub fn fit_bytes_by_owner(
        &self,
        depth: u64,
        banks: u64,
        history_owners: &[usize],
        recurrent_owners: &[usize],
        owner_count: usize,
    ) -> Result<Vec<u64>, String> {
        if history_owners.len() != self.domains.len()
            || recurrent_owners.len() != self.recurrent_components.len()
            || history_owners
                .iter()
                .chain(recurrent_owners)
                .any(|&owner| owner >= owner_count)
        {
            return Err("fit state owners differ from the planned domains and banks".into());
        }
        (0..owner_count)
            .map(|owner| {
                let history = self
                    .domains
                    .iter()
                    .zip(history_owners)
                    .filter(|(_, assigned)| **assigned == owner)
                    .try_fold(0u64, |total, (domain, _)| {
                        total
                            .checked_add(Self::history_domain_bytes_at_depth(
                                domain,
                                self.max_advance,
                                depth,
                            )?)
                            .ok_or_else(|| String::from("owner history fit charge overflows"))
                    })?;
                let components = self
                    .recurrent_components
                    .iter()
                    .zip(recurrent_owners)
                    .filter_map(|(component, &assigned)| (assigned == owner).then_some(component));
                let recurrent = self
                    .bank_slab_layout_for(components)?
                    .map_or(Ok(0), |layout| {
                        let slabs = banks.max(1).div_ceil(u64::from(self.bank_slab_banks()?));
                        layout
                            .slab_bytes
                            .checked_mul(slabs)
                            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
                            .ok_or_else(|| String::from("owner bank fit charge overflows"))
                    })?;
                history
                    .checked_add(recurrent)
                    .ok_or_else(|| "owner fit state charge overflows".into())
            })
            .collect()
    }

    pub fn bank_bytes_at_count(&self, banks: u64) -> Result<u64, String> {
        let Some(layout) = self.bank_slab_layout()? else {
            return Ok(0);
        };
        let slabs = banks.max(1).div_ceil(u64::from(self.bank_slab_banks()?));
        layout
            .slab_bytes
            .checked_mul(slabs)
            .and_then(|bytes| bytes.checked_add(layout.address_table_bytes))
            .ok_or_else(|| "bank slab fit charge overflows".into())
    }

    /// Address tables and the first history and bank slabs committed by
    /// `StateStore::new` for each present store.
    pub fn initial_committed_bytes(&self) -> Result<u64, String> {
        self.startup_history_bytes()?
            .checked_add(self.startup_bank_bytes()?)
            .ok_or_else(|| "initial state slab charge overflows".into())
    }

    /// Project physical startup charges by opened device, including alignment
    /// within each owner's recurrent slab. No allocations are made here.
    pub fn initial_committed_bytes_by_device(
        &self,
        history_devices: &[Rc<Device>],
        recurrent_devices: &[Rc<Device>],
    ) -> Result<Vec<(Rc<Device>, u64)>, String> {
        if history_devices.len() != self.domains.len()
            || recurrent_devices.len() != self.recurrent_components.len()
        {
            return Err(
                "state device assignments differ from the planned domains and banks".into(),
            );
        }
        let mut owners: Vec<Rc<Device>> = Vec::new();
        for device in history_devices.iter().chain(recurrent_devices) {
            if !owners.iter().any(|owner| owner.same_device(device)) {
                owners.push(device.clone());
            }
        }
        let indices = |devices: &[Rc<Device>]| {
            devices
                .iter()
                .map(|device| {
                    owners
                        .iter()
                        .position(|owner| owner.same_device(device))
                        .expect("known owner")
                })
                .collect::<Vec<_>>()
        };
        let bytes = self.initial_committed_bytes_by_owner(
            &indices(history_devices),
            &indices(recurrent_devices),
            owners.len(),
        )?;
        Ok(owners.into_iter().zip(bytes).collect())
    }

    /// Allocation-free projection for placement indices, before devices open.
    pub fn initial_committed_bytes_by_owner(
        &self,
        history_owners: &[usize],
        recurrent_owners: &[usize],
        owner_count: usize,
    ) -> Result<Vec<u64>, String> {
        if history_owners.len() != self.domains.len()
            || recurrent_owners.len() != self.recurrent_components.len()
            || history_owners
                .iter()
                .chain(recurrent_owners)
                .any(|&owner| owner >= owner_count)
        {
            return Err("state owners differ from the planned domains and banks".into());
        }
        (0..owner_count)
            .map(|owner| {
                let mut layouts = self
                    .domains
                    .iter()
                    .zip(history_owners)
                    .filter(|(_, assigned)| **assigned == owner)
                    .map(|(domain, _)| domain.slab_layout())
                    .collect::<Result<Vec<_>, _>>()?;
                let regions = self
                    .recurrent_components
                    .iter()
                    .zip(recurrent_owners)
                    .filter(|(_, assigned)| **assigned == owner)
                    .map(|(component, _)| SlabRegion {
                        element: Element::dense(component.dtype),
                        row_shape: component.shape.iter().map(|&size| size as u64).collect(),
                    })
                    .collect::<Vec<_>>();
                if !regions.is_empty() {
                    layouts.push(
                        SlabLayout::for_regions(
                            u64::from(self.bank_slab_banks()?),
                            self.bank_capacity
                                .storage_total()
                                .map_err(|error| error.to_string())?
                                as u64,
                            &regions,
                        )
                        .map_err(|error| error.to_string())?,
                    );
                }
                layouts
                    .into_iter()
                    .try_fold(0u64, |total, layout| {
                        total
                            .checked_add(layout.address_table_bytes)?
                            .checked_add(layout.slab_bytes)
                    })
                    .ok_or_else(|| "initial state slab charge overflows".into())
            })
            .collect()
    }

    /// The planned store and its one binding right.
    pub fn allocate(&self, device: Rc<Device>) -> Result<StoreBindings, String> {
        self.allocate_placed(
            self.domains.iter().map(|_| device.clone()).collect(),
            self.recurrent_components
                .iter()
                .map(|_| device.clone())
                .collect(),
        )
    }

    /// Allocate the planned domains on their owning devices, preserving the
    /// same allocation trace and transaction geometry as the admitted plan.
    pub fn allocate_placed(
        &self,
        history_devices: Vec<Rc<Device>>,
        recurrent_devices: Vec<Rc<Device>>,
    ) -> Result<StoreBindings, String> {
        let expected =
            self.initial_committed_bytes_by_device(&history_devices, &recurrent_devices)?;
        let (store, bindings) = StateStore::new_placed(
            history_devices,
            recurrent_devices,
            self.context_rows,
            self.max_advance,
            self.history.clone(),
            self.recurrent_components.clone(),
            self.bank_capacity,
        )
        .map_err(|error| error.to_string())?;
        let trace = store
            .allocation_trace()
            .map_err(|error| error.to_string())?;
        if trace.context_capacity != self.context_rows
            || trace.history
                != self
                    .domains
                    .iter()
                    .map(HistoryStorePlan::trace)
                    .collect::<Vec<_>>()
            || trace.bank_capacity != self.bank_capacity
            || trace.recurrent_bank_bytes != self.recurrent_bank_bytes
            || trace.zero_seed_bytes != self.zero_seed_bytes
            || trace.recurrent_pool_bytes != self.recurrent_pool_bytes
            || expected
                .iter()
                .any(|(device, bytes)| store.committed_bytes_on(device) != *bytes)
        {
            return Err("state allocation differs from its resource-plan projection".into());
        }
        Ok(bindings)
    }
}

impl ResourceBytes {
    pub fn total(self) -> Result<u64, String> {
        [
            self.target_weights,
            self.head_weights,
            self.vision_weights,
            self.history,
            self.recurrent_banks,
            self.prepared_programs,
            self.scratch,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| "resource byte total overflow".into())
    }
}

/// Immutable allocation authority. Every physical constructor receives the
/// relevant projection of this value rather than recomputing capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TargetBoundaryPlan {
    pub tensors: [Option<seismic::NativeTensorMetadata>; 4],
    pub slots: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HeadConditioningPlan {
    pub current: seismic::NativeTensorMetadata,
    pub previous: seismic::NativeTensorMetadata,
    pub slots: usize,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    boundary: Option<TargetBoundaryPlan>,
    conditioning: Option<HeadConditioningPlan>,
    decoder_range: Option<super::DecoderRange>,
    pub(super) domain_capacity_bytes: u64,
    /// Included in aggregate charges, but held outside the allocation domain.
    host_shared_bytes: u64,
    capacity: StateCapacityPlan,
    target_state: StateStorePlan,
    head_state: Option<StateStorePlan>,
    retained_entry_bytes: u64,
    bytes: ResourceBytes,
    pub(super) qualification_peak_bytes: u64,
    /// Independent pre-import phase of a placed load. Ordinary plans keep
    /// their existing conservative combined startup bound.
    preparation_peak_bytes: u64,
    target_graph: NativeGraphCharge,
    target_readout_graph: NativeGraphCharge,
    head_graph: Option<NativeGraphCharge>,
    vision_graph: Option<NativeGraphCharge>,
    state_graph: NativeGraphCharge,
    steady_committed_bytes: u64,
    startup_peak_bytes: u64,
}

impl ResourcePlan {
    pub(crate) fn boundary(&self) -> Option<&TargetBoundaryPlan> { self.boundary.as_ref() }
    pub(crate) fn conditioning(&self) -> Option<&HeadConditioningPlan> { self.conditioning.as_ref() }

    /// Admit the second owner's persistent transfer slots using the sealed
    /// source outputs and destination input. Revalidate the full startup peak.
    pub fn with_boundary_from(mut self, source: &PreparedTargetGraphs, destination: &PreparedTargetGraphs) -> Result<Self, String> {
        let range = self.decoder_range.as_ref().ok_or("boundary requires range resources")?;
        if self.boundary.is_some() || range.blocks.start == 0
            || source.owned_blocks().end != range.blocks.start
            || destination.owned_blocks() != range.blocks {
            return Err("boundary graphs differ from admitted adjacent ranges".into());
        }
        let tensors = source.outgoing_boundary()?;
        if tensors[0].as_ref() != Some(&destination.incoming_hidden()?) {
            return Err("boundary producer and consumer descriptors disagree".into());
        }
        let bytes = tensors.iter().flatten().try_fold(0u64, |sum, tensor| sum.checked_add(tensor.canonical_bytes))
            .and_then(|bytes| bytes.checked_mul(self.target_graph.activations as u64))
            .ok_or("boundary byte charge overflows")?;
        self.bytes.scratch = self.bytes.scratch.checked_add(bytes).ok_or("boundary scratch charge overflows")?;
        self.boundary = Some(TargetBoundaryPlan { tensors, slots: self.target_graph.activations, bytes });
        self.validate()
    }

    pub fn decoder_range(&self) -> Option<&super::DecoderRange> {
        self.decoder_range.as_ref()
    }

    pub(crate) fn qualification_peak_bytes(&self) -> u64 {
        self.qualification_peak_bytes
    }
    pub fn domain_capacity_bytes(&self) -> u64 {
        self.domain_capacity_bytes
    }
    pub fn capacity(&self) -> StateCapacityPlan {
        self.capacity
    }
    pub fn target_state(&self) -> &StateStorePlan {
        &self.target_state
    }
    pub fn head_state(&self) -> Option<&StateStorePlan> {
        self.head_state.as_ref()
    }
    pub fn retained_entry_bytes(&self) -> u64 {
        self.retained_entry_bytes
    }
    pub fn bytes(&self) -> ResourceBytes {
        self.bytes
    }

    /// Every family's charge.
    fn graph_charges(&self) -> Vec<&NativeGraphCharge> {
        [
            Some(&self.target_graph),
            Some(&self.target_readout_graph),
            self.head_graph.as_ref(),
            self.vision_graph.as_ref(),
            Some(&self.state_graph),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// The device's one workspace arena, which every family binds.
    pub fn arena_bytes(&self) -> u64 {
        arena_bytes(&self.graph_charges())
    }

    /// Persistent upload storage in the backend's shared-storage domain.
    pub fn shared_upload_bytes(&self) -> Result<u64, String> {
        graph_shared_upload_bytes(&self.graph_charges())
    }

    pub(super) fn validate(mut self) -> Result<Self, String> {
        let scratch = graph_scratch_bytes(&self.graph_charges())?
            .checked_add(self.boundary.as_ref().map_or(0, |boundary| boundary.bytes))
            .and_then(|bytes| bytes.checked_add(self.conditioning.as_ref().map_or(0, |plan| plan.bytes)))
            .ok_or("boundary scratch charge overflows")?;
        if self.bytes.scratch != scratch {
            return Err(
                "resource plan pooled byte charge differs from admitted Seismic footprints".into(),
            );
        }
        self.steady_committed_bytes = self.bytes.total()?;
        let loading_peak = if self.decoder_range.is_some() {
            let state = self.bytes.history.checked_add(self.bytes.recurrent_banks)
                .ok_or("range initial state bytes overflow")?;
            let import_and_state = if self.head_state.is_some() {
                state.checked_add(self.qualification_peak_bytes)
                    .ok_or("head import and initial state charge overflow")?
            } else {
                state.max(self.qualification_peak_bytes)
            };
            self.steady_committed_bytes.checked_sub(state)
                .and_then(|bytes| bytes.checked_add(import_and_state))
                .ok_or("range loading peak byte count overflow")?
        } else {
            self.steady_committed_bytes.checked_add(self.qualification_peak_bytes)
                .ok_or("startup peak byte count overflow")?
        };
        self.startup_peak_bytes = loading_peak.max(self.preparation_peak_bytes);
        let allocation_peak = loading_peak
            .checked_sub(self.host_shared_bytes)
            .ok_or("shared storage exceeds startup charge")?
            .max(self.preparation_peak_bytes);
        if allocation_peak > self.domain_capacity_bytes {
            return Err(format!(
                "resource plan startup peak {allocation_peak} exceeds {} bytes",
                self.domain_capacity_bytes
            ));
        }
        Ok(self)
    }

    pub fn target_graph(&self) -> NativeGraphCharge {
        self.target_graph.clone()
    }
    pub fn target_readout_graph(&self) -> NativeGraphCharge {
        self.target_readout_graph.clone()
    }
    pub fn head_graph(&self) -> Option<NativeGraphCharge> {
        self.head_graph.clone()
    }
    pub fn vision_graph(&self) -> Option<NativeGraphCharge> {
        self.vision_graph.clone()
    }
    pub fn state_graph(&self) -> NativeGraphCharge {
        self.state_graph.clone()
    }
    pub fn steady_committed_bytes(&self) -> u64 {
        self.steady_committed_bytes
    }
    pub fn startup_peak_bytes(&self) -> u64 {
        self.startup_peak_bytes
    }

    pub fn allocate_target_state(&self, device: Rc<Device>) -> Result<StoreBindings, String> {
        if self.decoder_range.is_some() {
            return Err("range resources require coordinated placed-state allocation".into());
        }
        self.target_state.allocate(device)
    }

    pub fn allocate_head_state(&self, device: Rc<Device>) -> Result<Option<StoreBindings>, String> {
        self.head_state
            .as_ref()
            .map(|plan| plan.allocate(device))
            .transpose()
    }
}

pub struct ResourcePlanner;

/// State shape and retention capacity are known before Seismic constructs
/// graph ports. Graph scratch is admitted only after Seismic seals its checked
/// stage topology and reports the exact storage footprint.
#[derive(Clone, Debug, PartialEq)]
pub struct StateResourcePlan {
    placement: Option<super::DecoderPlacement>,
    owner_capacities: Vec<ResourceCapacity>,
    history_owners: Vec<usize>,
    recurrent_owners: Vec<usize>,
    definition: ModelDefinition,
    load: ModelLoadPlan,
    codec: KvCodec,
    limits: ResourceLimits,
    capacity_bytes: ResourceCapacity,
    capacity: StateCapacityPlan,
    target_state: StateStorePlan,
    head_state: Option<StateStorePlan>,
    retained_entry_bytes: u64,
    history_bytes: u64,
    recurrent_banks_bytes: u64,
}

impl StateResourcePlan {
    pub fn placement(&self) -> Option<&super::DecoderPlacement> {
        self.placement.as_ref()
    }

    pub fn history_owners(&self) -> &[usize] {
        &self.history_owners
    }

    pub fn recurrent_owners(&self) -> &[usize] {
        &self.recurrent_owners
    }

    pub fn startup_state_bytes_by_owner(&self) -> Result<Vec<u64>, String> {
        let placement = self
            .placement
            .as_ref()
            .ok_or("state has no decoder device placement")?;
        let mut bytes = self.target_state.initial_committed_bytes_by_owner(
            &self.history_owners,
            &self.recurrent_owners,
            placement.ranges().len(),
        )?;
        bytes[0] = bytes[0].checked_add(self.head_state.as_ref()
            .map(|state| state.initial_committed_bytes()).transpose()?.unwrap_or(0))
            .ok_or("placed head startup charge overflows")?;
        Ok(bytes)
    }

    /// Materialize the already projected domains on the exact selected opened
    /// devices. Identity and all startup projections are checked before allocation.
    pub fn allocate_target_state_placed(
        &self,
        devices: &[Rc<Device>],
    ) -> Result<StoreBindings, String> {
        let placement = self
            .placement
            .as_ref()
            .ok_or("state has no decoder device placement")?;
        if devices.len() != placement.ranges().len()
            || devices
                .iter()
                .zip(placement.ranges())
                .any(|(device, range)| device.info().selector != range.device)
        {
            return Err("opened state devices differ from the admitted decoder placement".into());
        }
        self.startup_state_bytes_by_owner()?;
        let history = self
            .history_owners
            .iter()
            .map(|&owner| devices[owner].clone())
            .collect();
        let recurrent = self
            .recurrent_owners
            .iter()
            .map(|&owner| devices[owner].clone())
            .collect();
        // allocate_placed checks actual backing per opened device, including
        // recurrent group alignment, against this same projection.
        let bindings = self.target_state.allocate_placed(history, recurrent)?;
        Ok(bindings)
    }

    pub fn limits(&self) -> ResourceLimits {
        self.limits
    }

    /// Whether the planned device forms Metal tensor operations
    /// (`ResourceCapacity::tensor_operations`, which its opened device must
    /// confirm).
    pub fn tensor_operations(&self) -> bool {
        self.capacity_bytes.tensor_operations == TensorOperations::Formed
    }

    /// Whether prefill attention graphs of `head_width`-column heads over
    /// this plan's history come in classes that list the history row tiles
    /// their launch's rows see, beside the class that lists none: where the
    /// planned device has a form of the affine prefill entry that reads
    /// history decoded for the call ([`reads_decoded_history`]).
    pub fn lists_history_tiles(&self, backend: BackendName, head_width: u64) -> bool {
        self.decoded_history_form(backend, head_width).is_some()
    }

    /// The form of the affine prefill entry that reads this plan's history
    /// decoded for `head_width`-column heads on the planned device of
    /// `backend`: what the listing classes of such a block's graphs follow
    /// ([`HistoryStorePlan::listed_tile_classes`]). None where the graphs
    /// list no tiles.
    pub fn decoded_history_form(
        &self,
        backend: BackendName,
        head_width: u64,
    ) -> Option<DecodedHistoryForm> {
        DecodedHistoryForm::of(backend, self.tensor_operations(), head_width)
            .filter(|_| self.codec == KvCodec::AffineK8V4)
    }

    pub fn capacity(&self) -> StateCapacityPlan {
        self.capacity
    }

    pub fn fit_state_bytes(&self, depth: u64, banks: u64) -> Result<u64, String> {
        if self.placement.is_some() {
            return Err("distributed state fit requires per-device charges".into());
        }
        let target_history = self.target_state.history_bytes_at_depth(depth)?;
        let head_history = self
            .head_state
            .as_ref()
            .map(|state| state.history_bytes_at_depth(depth))
            .transpose()?
            .unwrap_or(0);
        let target_banks = self.target_state.bank_bytes_at_count(banks)?;
        target_history
            .checked_add(head_history)
            .and_then(|bytes| bytes.checked_add(target_banks))
            .ok_or_else(|| "state fit charge overflows".into())
    }

    pub fn fit_state_bytes_by_owner(&self, depth: u64, banks: u64) -> Result<Vec<u64>, String> {
        let placement = self
            .placement
            .as_ref()
            .ok_or("per-owner fit requires placed state")?;
        let mut bytes = self.target_state.fit_bytes_by_owner(
            depth,
            banks,
            &self.history_owners,
            &self.recurrent_owners,
            placement.ranges().len(),
        )?;
        bytes[0] = bytes[0].checked_add(self.head_state.as_ref()
            .map(|state| state.history_bytes_at_depth(depth)).transpose()?.unwrap_or(0))
            .ok_or("placed head fit charge overflows")?;
        Ok(bytes)
    }

    /// The state slabs a load commits at startup, before any request.
    pub fn startup_state_bytes(&self) -> Result<u64, String> {
        self.history_bytes
            .checked_add(self.recurrent_banks_bytes)
            .ok_or_else(|| "startup state bytes overflow".into())
    }

    pub fn target_state(&self) -> &StateStorePlan {
        &self.target_state
    }

    pub fn head_state(&self) -> Option<&StateStorePlan> {
        self.head_state.as_ref()
    }
}

impl ResourcePlanner {
    pub fn state_plan(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        capacity_bytes: ResourceCapacity,
    ) -> Result<StateResourcePlan, String> {
        Self::state_plan_with_placement(
            definition,
            load,
            method,
            codec,
            limits,
            capacity_bytes,
            None,
        )
    }

    /// One transaction's state geometry across decoder devices. Logical address
    /// ceilings follow each owner's capacity and state cost; physical charges
    /// remain separate and are never admitted against their sum.
    pub fn state_plan_placed(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        placement: &super::DecoderPlacement,
        capacities: &[ResourceCapacity],
    ) -> Result<StateResourcePlan, String> {
        if capacities.len() != placement.ranges().len()
            || capacities.is_empty()
            || capacities.iter().any(|capacity| capacity.domain_bytes == 0)
        {
            return Err("state placement requires one positive capacity per device".into());
        }
        if placement.ranges().last().map(|range| range.blocks.end)
            != Some(definition.decoder.blocks.len())
        {
            return Err("state placement differs from the model's decoder extent".into());
        }
        if load.vision.is_some()
            || !matches!(method, PlannedMethod::Plain | PlannedMethod::Mtp { .. })
            || load.head.is_some() && !definition.head.as_ref().is_some_and(|head|
                head.history == magnitude_family_contracts::HeadHistory::Own) {
            return Err("placed drafting requires a causal head with its own history".into());
        }
        if capacities
            .iter()
            .any(|capacity| capacity.tensor_operations != capacities[0].tensor_operations)
        {
            return Err("distributed state requires compatible history operator forms".into());
        }
        let capacity = ResourceCapacity {
            domain_bytes: capacities
                .iter()
                .map(|capacity| capacity.domain_bytes)
                .min()
                .unwrap(),
            tensor_operations: capacities[0].tensor_operations,
        };
        Self::state_plan_with_placement(
            definition,
            load,
            method,
            codec,
            limits,
            capacity,
            Some((placement.clone(), capacities.to_vec())),
        )
    }

    fn state_plan_with_placement(
        definition: &ModelDefinition,
        load: &ModelLoadPlan,
        method: PlannedMethod,
        codec: KvCodec,
        limits: ResourceLimits,
        capacity_bytes: ResourceCapacity,
        placement: Option<(super::DecoderPlacement, Vec<ResourceCapacity>)>,
    ) -> Result<StateResourcePlan, String> {
        let (placement, owner_capacities) = match placement {
            Some((placement, capacities)) => (Some(placement), capacities),
            None => (None, vec![capacity_bytes]),
        };
        if limits.max_launch_slots == 0
            || limits.max_launch_slots > limits.max_launch_rows
            || limits.max_launch_rows == 0
            || limits.max_selected_rows == 0
            || limits.max_selected_rows > limits.max_launch_rows
            || limits.max_drafting_slots == 0
            || limits.max_drafting_slots > limits.max_launch_slots
            || limits.exported_logits_rows > limits.max_launch_rows
            || limits.max_images_per_request == 0
        {
            return Err("resource limits must be positive".into());
        }
        if limits.max_images_per_request > magnitude_artifacts::MAX_IMAGES_PER_REQUEST {
            return Err("planned image limit exceeds preprocessing capacity".into());
        }
        if capacity_bytes.domain_bytes == 0 {
            return Err("device domain has zero capacity".into());
        }
        let drafter = if load.head.is_some() {
            crate::operators::draft::drafter_blocks(definition)
        } else {
            Vec::new()
        };
        let layout =
            ModelStateLayout::derive(&definition.decoder, &drafter, codec, method.draft_rows())?;
        let (layout, history_owners, recurrent_owners) = match &placement {
            Some(placement) => {
                let placed = placement.place_state(&layout)?;
                (
                    placed.layout,
                    placed.history_owners,
                    placed.recurrent_owners,
                )
            }
            None => {
                let history = vec![
                    0;
                    layout
                        .target_history
                        .iter()
                        .filter(|domain| !matches!(domain, HistoryDomainLayout::Shared { .. }))
                        .count()
                ];
                let recurrent = vec![0; layout.target_recurrent.len()];
                (layout, history, recurrent)
            }
        };
        let history_domains = || layout.target_history.iter().chain(&layout.head_history);
        let checkpoint_history_row_bytes = history_row_bytes(&layout.target_history)?
            .checked_add(history_row_bytes(&layout.head_history)?)
            .ok_or("history row byte count overflow")?;
        let recurrent_bank_bytes = recurrent_bank_bytes(&layout.target_recurrent)?;
        let context = usize::try_from(definition.decoder.context_limit)
            .map_err(|_| "context limit exceeds host domain")?;
        let max_advance = limits.max_launch_rows;
        // A checkpoint's window rows: `n` per Window(n) domain.
        let checkpoint_window_bytes = history_domains()
            .filter(|domain| matches!(domain.kind(), HistoryDomainKind::Window { .. }))
            .try_fold(0u64, |total, domain| {
                let rows = domain
                    .kind()
                    .checkpoint_rows(context)
                    .map_err(|error| error.to_string())?;
                domain
                    .row_bytes()
                    .map_err(|error| error.to_string())?
                    .checked_mul(rows as u64)
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(|| String::from("window checkpoint byte count overflow"))
            })?;
        // Generation methods keep their carried feature rows on the host, so
        // a retained entry charges only numerical state. Entries on one path
        // share their history rows, so an entry's marginal state is its recurrent
        // bank; without recurrent state it is a full context of history.
        let retained_entry_bytes = if recurrent_bank_bytes != 0 {
            recurrent_bank_bytes
        } else {
            checkpoint_history_row_bytes
                .checked_mul(definition.decoder.context_limit)
                .and_then(|bytes| bytes.checked_add(checkpoint_window_bytes))
                .ok_or("checkpoint state byte count overflow")?
        };
        // One row in every stored domain.
        let request_row_bytes = history_domains().try_fold(0u64, |total, domain| {
            total
                .checked_add(domain.row_bytes().map_err(|error| error.to_string())?)
                .ok_or_else(|| String::from("history row byte count overflow"))
        })?;
        // Reservations are index bounds sealed into graphs, not memory: every
        // bank and history row is committed on demand under a heap claim. A
        // live request holds its accepted bank and, in flight, a successor
        // (two with lookahead); a branch checkpoint or retention entry holds
        // one more. Retention is bounded by what the domain could back.
        let retention_entries = if retained_entry_bytes == 0 {
            0
        } else {
            usize::try_from(capacity_bytes.domain_bytes / retained_entry_bytes)
                .unwrap_or(usize::MAX)
        };
        let history_bound =
            usize::try_from(capacity_bytes.domain_bytes / checkpoint_history_row_bytes.max(1))
                .unwrap_or(usize::MAX);
        // A request needs at least one history row and one recurrent bank.
        // This is an address-space ceiling from physical bytes, never an
        // admission policy; each actual row and bank is claimed on demand.
        let minimum_request_bytes = request_row_bytes
            .checked_add(recurrent_bank_bytes)
            .ok_or("minimum request byte count overflow")?
            .max(1);
        let possible_requests =
            usize::try_from(capacity_bytes.domain_bytes / minimum_request_bytes)
                .unwrap_or(usize::MAX)
                .max(1);
        // A distributed address ceiling is the most constrained owner's
        // capacity divided by that owner's own state cost. Dividing one GPU's
        // capacity by the whole model's state would incorrectly reject a
        // context that fits when its history is split over both GPUs.
        let (retention_entries, history_bound, possible_requests) = if placement.is_some() {
            let mut retained = None::<usize>;
            let mut history = None::<usize>;
            let mut requests = None::<usize>;
            for (owner, capacity) in owner_capacities.iter().enumerate() {
                let mut domains = layout
                    .target_history
                    .iter()
                    .filter(|domain| !matches!(domain, HistoryDomainLayout::Shared { .. }))
                    .zip(&history_owners)
                    .filter(|(_, assigned)| **assigned == owner)
                    .map(|(domain, _)| domain.clone())
                    .collect::<Vec<_>>();
                if owner == 0 { domains.extend(layout.head_history.iter().cloned()); }
                let components = layout
                    .target_recurrent
                    .iter()
                    .zip(&recurrent_owners)
                    .filter(|(_, assigned)| **assigned == owner)
                    .map(|(component, _)| component.clone())
                    .collect::<Vec<_>>();
                let history_row = history_row_bytes(&domains)?;
                let bank = self::recurrent_bank_bytes(&components)?;
                let request_row = domains.iter().try_fold(0u64, |bytes, domain| {
                    bytes
                        .checked_add(domain.row_bytes().map_err(|error| error.to_string())?)
                        .ok_or_else(|| "placed request row bytes overflow".to_owned())
                })?;
                let window = domains
                    .iter()
                    .filter(|domain| matches!(domain.kind(), HistoryDomainKind::Window { .. }))
                    .try_fold(0u64, |bytes, domain| {
                        let rows = domain
                            .kind()
                            .checkpoint_rows(context)
                            .map_err(|error| error.to_string())?;
                        domain
                            .row_bytes()
                            .map_err(|error| error.to_string())?
                            .checked_mul(rows as u64)
                            .and_then(|size| bytes.checked_add(size))
                            .ok_or_else(|| "placed checkpoint window bytes overflow".to_owned())
                    })?;
                let checkpoint = if recurrent_bank_bytes != 0 {
                    bank
                } else {
                    history_row
                        .checked_mul(definition.decoder.context_limit)
                        .and_then(|size| size.checked_add(window))
                        .ok_or("placed checkpoint bytes overflow")?
                };
                let constrain = |limit: &mut Option<usize>, bytes: u64| {
                    if bytes != 0 {
                        let count =
                            usize::try_from(capacity.domain_bytes / bytes).unwrap_or(usize::MAX);
                        *limit = Some(limit.map_or(count, |prior| prior.min(count)));
                    }
                };
                constrain(&mut retained, checkpoint);
                constrain(&mut history, history_row);
                constrain(
                    &mut requests,
                    request_row
                        .checked_add(bank)
                        .ok_or("placed request bytes overflow")?,
                );
            }
            (
                retained.unwrap_or(0),
                history.unwrap_or(history_bound),
                requests.unwrap_or(possible_requests).max(1),
            )
        } else {
            (retention_entries, history_bound, possible_requests)
        };
        if checkpoint_history_row_bytes != 0 && history_bound < context {
            return Err("one context exceeds an owning device's history capacity".into());
        }
        let bank_capacity = BankCapacity {
            active: possible_requests,
            in_flight: possible_requests
                .checked_mul(limits.longest_chain())
                .ok_or("in-flight bank count overflow")?,
            retained: possible_requests
                .checked_add(retention_entries)
                .ok_or("retained numerical capacity overflow")?,
        };
        let capacity = StateCapacityPlan {
            live_requests: possible_requests,
            checkpoints: possible_requests,
            retention_entries,
            // The history reservation (graphs are sealed over it): a context
            // for every bank owner, but never more rows than the device's
            // stable domain capacity could ever back.
            history_rows: history_bound.max(context),
        };
        let bank_count = bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?;
        let recurrent_pool_bytes = recurrent_bank_bytes
            .checked_mul(u64::try_from(bank_count).map_err(|_| "bank capacity exceeds u64")?)
            .ok_or("recurrent bank allocation byte count overflow")?;
        let rows = HistoryRows {
            context,
            max_advance,
            token: capacity.history_rows,
            domain_bytes: capacity_bytes.domain_bytes,
        };
        let target_state = state_store_plan(
            rows,
            layout.target_history,
            layout.target_recurrent,
            bank_capacity,
        )?;
        let head_state = (!layout.head_history.is_empty())
            .then(|| state_store_plan(rows, layout.head_history, Vec::new(), bank_capacity))
            .transpose()?;
        let planned_recurrent = target_state
            .recurrent_pool_bytes
            .checked_add(
                head_state
                    .as_ref()
                    .map_or(0, |state| state.recurrent_pool_bytes),
            )
            .ok_or("recurrent allocation byte count overflow")?;
        if planned_recurrent != recurrent_pool_bytes {
            return Err("state projections disagree on recurrent allocation bytes".into());
        }
        // Seismic's fixed address tables and first history and bank slabs are
        // committed at store construction. Further slabs use heap claims.
        let history = target_state
            .startup_history_bytes()?
            .checked_add(
                head_state
                    .as_ref()
                    .map(|state| state.startup_history_bytes())
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("startup history slab charge overflows")?;
        let planned_recurrent = target_state
            .startup_bank_bytes()?
            .checked_add(
                head_state
                    .as_ref()
                    .map(|state| state.startup_bank_bytes())
                    .transpose()?
                    .unwrap_or(0),
            )
            .ok_or("startup bank slab charge overflows")?;
        let planned_recurrent = if let Some(placement) = &placement {
            let startup = target_state
                .initial_committed_bytes_by_owner(
                    &history_owners,
                    &recurrent_owners,
                    placement.ranges().len(),
                )?
                .into_iter()
                .try_fold(0u64, |total, bytes| total.checked_add(bytes))
                .ok_or("distributed state charge overflows")?;
            startup.checked_add(head_state.as_ref()
                .map(|state| state.initial_committed_bytes()).transpose()?.unwrap_or(0))
                .ok_or("placed head startup charge overflows")?
                .checked_sub(history)
                .ok_or("history charge exceeds distributed startup")?
        } else {
            planned_recurrent
        };
        Ok(StateResourcePlan {
            placement,
            owner_capacities,
            history_owners,
            recurrent_owners,
            definition: definition.clone(),
            load: load.clone(),
            codec,
            limits,
            capacity_bytes,
            capacity,
            target_state,
            head_state,
            retained_entry_bytes,
            history_bytes: history,
            recurrent_banks_bytes: planned_recurrent,
        })
    }

    /// Admit one physical owner using its exact local inventory and sealed
    /// graphs. State remains one coordinated store; this plan cannot allocate it.
    pub fn plan_range_with_state(
        backend: BackendName,
        shared_role: crate::platform::DomainRole,
        state: &StateResourcePlan,
        load: &super::DecoderRangeLoadPlan,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedDrafterGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<ResourcePlan, String> {
        let placement = state
            .placement
            .as_ref()
            .ok_or("range admission requires placed state")?;
        let owner = placement
            .ranges()
            .iter()
            .position(|range| range == load.range())
            .ok_or("range inventory differs from state placement")?;
        if !load.shares_model_inventory(&state.load)
            || load.decoder_blocks() != state.definition.decoder.blocks.len()
            || target_graphs.owned_blocks() != load.range().blocks
        {
            return Err("range graph or decoder extent differs from placement".into());
        }
        let startup = state.startup_state_bytes_by_owner()?[owner];
        let head_state = (owner == 0).then(|| state.head_state.clone()).flatten();
        if head_graphs.is_some() != head_state.is_some() || head_graphs.is_some() != load.physical_load().head.is_some() {
            return Err("range head graphs, state and physical inventory disagree".into());
        }
        let history = state
            .target_state
            .domains
            .iter()
            .zip(&state.history_owners)
            .filter(|(_, assigned)| **assigned == owner)
            .try_fold(0u64, |total, (domain, _)| {
                let layout = domain.slab_layout()?;
                total
                    .checked_add(layout.address_table_bytes)
                    .and_then(|bytes| bytes.checked_add(layout.slab_bytes))
                    .ok_or_else(|| String::from("range history charge overflows"))
            })?.checked_add(head_state.as_ref().map(|state| state.startup_history_bytes())
                .transpose()?.unwrap_or(0)).ok_or("range head history charge overflows")?;
        let slots = state.limits.startup_slots();
        let target_graph = NativeGraphCharge::from_prepared(target_graphs, slots.target)?;
        // Entry/readout are still prepared on each owner and must be charged.
        let target_readout_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(target_readout_graphs.family()),
            slots.readout,
        )?;
        let state_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(state_graphs.family()),
            slots.state,
        )?;
        let head_graph = head_graphs.map(|graphs| NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(graphs.family()), slots.head)).transpose()?;
        let conditioning = head_graphs.map(|graphs| {
            let (current, previous) = graphs.conditioning_transfer_metadata()?;
            let bytes = current.canonical_bytes.checked_add(previous.canonical_bytes)
                .and_then(|bytes| bytes.checked_mul(slots.head.activations as u64))
                .ok_or("head conditioning transfer charge overflows")?;
            Ok::<_, String>(HeadConditioningPlan { current, previous, slots: slots.head.activations, bytes })
        }).transpose()?;
        let charges = [Some(&target_graph), Some(&target_readout_graph), head_graph.as_ref(), Some(&state_graph)]
            .into_iter().flatten().collect::<Vec<_>>();
        let prepared_programs = crate::AttestedPrograms::planned_invocation_workspace_bytes(
            &load
                .graph_load()
                .program_plan(&state.definition, state.codec)
                .map_err(|error| error.to_string())?,
            backend,
            state.limits,
            state.owner_capacities[owner].tensor_operations == TensorOperations::Formed,
        )
        .map_err(|error| error.to_string())?;
        let bytes = ResourceBytes {
            target_weights: load.weight_bytes(),
            head_weights: weight_bytes_by_component(load.physical_load())?[1],
            vision_weights: 0,
            history,
            recurrent_banks: startup
                .checked_sub(history)
                .ok_or("range history exceeds startup state")?,
            prepared_programs,
            scratch: graph_scratch_bytes(&charges)?.checked_add(conditioning.as_ref().map_or(0, |plan| plan.bytes))
                .ok_or("range conditioning scratch charge overflows")?,
        };
        let host_shared_bytes = match shared_role {
            crate::platform::DomainRole::Allocation => 0,
            crate::platform::DomainRole::Staging => load.physical_load().shared_weight_bytes()?
                .checked_add(graph_shared_upload_bytes(&charges)?)
                .ok_or("range shared storage charge overflows")?,
        };
        ResourcePlan {
            boundary: None,
            conditioning,
            decoder_range: Some(load.range().clone()),
            domain_capacity_bytes: state.owner_capacities[owner].domain_bytes,
            host_shared_bytes,
            capacity: state.capacity,
            target_state: state.target_state.clone(),
            head_state,
            retained_entry_bytes: state.retained_entry_bytes,
            bytes,
            qualification_peak_bytes: load.import_peak_bytes(),
            preparation_peak_bytes: load.preparation_peak_bytes(
                &state.definition,
                state.codec,
                backend,
                state.limits,
                state.owner_capacities[owner].tensor_operations == TensorOperations::Formed,
            )?,
            target_graph,
            target_readout_graph,
            head_graph,
            vision_graph: None,
            state_graph,
            steady_committed_bytes: 0,
            startup_peak_bytes: 0,
        }
        .validate()
    }

    pub fn plan_with_state(
        backend: BackendName,
        shared_role: crate::platform::DomainRole,
        state: StateResourcePlan,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedDrafterGraphs>,
        vision_graphs: Option<&PreparedVisionGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<ResourcePlan, String> {
        if state.placement.is_some() {
            return Err("distributed state requires per-device resource admission".into());
        }
        let definition = &state.definition;
        let load = &state.load;
        let limits = state.limits;
        let capacity_bytes = state.capacity_bytes;
        let qualification_peak = crate::AttestedPrograms::qualification_peak_bytes(load)?;
        let source_import_peak = load
            .target
            .iter()
            .chain(load.head.iter().flatten())
            .chain(load.vision.iter().flatten())
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        let source_import_peak = source_import_peak_bytes(source_import_peak)?;
        let qualification_peak_bytes = qualification_peak.max(source_import_peak);
        let slots = limits.startup_slots();
        let target_graph = NativeGraphCharge::from_prepared(target_graphs, slots.target)?;
        let target_readout_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(target_readout_graphs.family()),
            slots.readout,
        )?;
        let head_graph = head_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    slots.head,
                )
            })
            .transpose()?;
        let vision_graph = vision_graphs
            .map(|graphs| {
                NativeGraphCharge::from_footprint(
                    FamilyFootprint::serial(graphs.family()),
                    slots.vision,
                )
            })
            .transpose()?;
        let state_graph = NativeGraphCharge::from_footprint(
            FamilyFootprint::serial(state_graphs.family()),
            slots.state,
        )?;
        let charges = [
            Some(&target_graph),
            Some(&target_readout_graph),
            head_graph.as_ref(),
            vision_graph.as_ref(),
            Some(&state_graph),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let scratch = graph_scratch_bytes(&charges)?;
        let prepared_programs = crate::AttestedPrograms::planned_invocation_workspace_bytes(
            &load
                .program_plan(definition, state.codec)
                .map_err(|error| error.to_string())?,
            backend,
            state.limits,
            state.tensor_operations(),
        )
        .map_err(|error| error.to_string())?;
        let [target_weights, head_weights, vision_weights] = weight_bytes_by_component(load)?;
        let bytes = ResourceBytes {
            target_weights,
            head_weights,
            vision_weights,
            history: state.history_bytes,
            recurrent_banks: state.recurrent_banks_bytes,
            prepared_programs,
            scratch,
        };
        let host_shared_bytes = match shared_role {
            crate::platform::DomainRole::Allocation => 0,
            crate::platform::DomainRole::Staging => load.shared_weight_bytes()?
                .checked_add(graph_shared_upload_bytes(&charges)?)
                .ok_or("shared storage charge overflows")?,
        };
        let required = bytes
            .total()?
            .checked_sub(host_shared_bytes)
            .ok_or("shared storage exceeds resource charge")?;
        if std::env::var_os("MAGNITUDE_TRACE_RESOURCES").is_some() {
            eprintln!(
                "resource plan domain_capacity={} required={} qualification_peak={} weights=[{},{},{}] history={} recurrent_banks={} prepared_programs={} scratch={} graph=[target:{},readout:{},head:{},vision:{},state:{}]",
                capacity_bytes.domain_bytes,
                required,
                qualification_peak_bytes,
                bytes.target_weights,
                bytes.head_weights,
                bytes.vision_weights,
                bytes.history,
                bytes.recurrent_banks,
                bytes.prepared_programs,
                bytes.scratch,
                target_graph.committed_bytes,
                target_readout_graph.committed_bytes,
                head_graph.as_ref().map_or(0, |graph| graph.committed_bytes),
                vision_graph.as_ref().map_or(0, |graph| graph.committed_bytes),
                state_graph.committed_bytes,
            );
            // Each family's placed workspace against its liveness floor: a
            // placement regression shows as placed above floor.
            let workspace = |family: &seismic::NativeGraphFamily| {
                format!(
                    "{}/{}",
                    family.workspace_bytes(),
                    family.workspace_floor_bytes()
                )
            };
            eprintln!(
                "resource plan arena={} workspace placed/floor=[target:{},readout:{},head:{},vision:{},state:{}]",
                arena_bytes(&charges),
                workspace(target_graphs.family()),
                workspace(target_readout_graphs.family()),
                head_graphs.map_or_else(|| "-".into(), |graphs| workspace(graphs.family())),
                vision_graphs.map_or_else(|| "-".into(), |graphs| workspace(graphs.family())),
                workspace(state_graphs.family()),
            );
        }
        if required > capacity_bytes.domain_bytes {
            return Err(format!(
                "resource plan requires {required} bytes but the device domain has {}",
                capacity_bytes.domain_bytes
            ));
        }
        ResourcePlan {
            boundary: None,
            conditioning: None,
            decoder_range: None,
            domain_capacity_bytes: capacity_bytes.domain_bytes,
            host_shared_bytes,
            capacity: state.capacity,
            target_state: state.target_state,
            head_state: state.head_state,
            retained_entry_bytes: state.retained_entry_bytes,
            bytes,
            qualification_peak_bytes,
            preparation_peak_bytes: 0,
            target_graph,
            target_readout_graph,
            head_graph,
            vision_graph,
            state_graph,
            steady_committed_bytes: 0,
            startup_peak_bytes: 0,
        }
        .validate()
    }
}

/// The row reservations of a store's history domains.
#[derive(Clone, Copy)]
struct HistoryRows {
    context: usize,
    max_advance: usize,
    /// Every Token domain's reservation.
    token: usize,
    /// A Window domain reserves what the device domain could back, and at
    /// least its row limit.
    domain_bytes: u64,
}

impl HistoryRows {
    fn rows(self, layout: &HistoryDomainLayout) -> Result<usize, String> {
        let kind = layout.kind();
        Ok(match kind {
            HistoryDomainKind::Token => self.token,
            HistoryDomainKind::Window { .. } => {
                let row_bytes = layout.row_bytes().map_err(|error| error.to_string())?;
                usize::try_from(self.domain_bytes / row_bytes.max(1))
                    .unwrap_or(usize::MAX)
                    .max(
                        kind.row_limit(self.context, self.max_advance)
                            .map_err(|error| error.to_string())?,
                    )
            }
            HistoryDomainKind::Shared { .. } => 0,
            // A block row per `rate` of the Token domain's.
            HistoryDomainKind::Block { rate } => self.token.div_ceil(rate),
        })
    }
}

fn state_store_plan(
    rows: HistoryRows,
    history: Vec<HistoryDomainLayout>,
    recurrent_components: Vec<ComponentSpec>,
    bank_capacity: BankCapacity,
) -> Result<StateStorePlan, String> {
    let layouts = history;
    // The rows the store reserves for each stored domain (whole pages), which
    // graphs are sealed over.
    let history = layouts
        .iter()
        .map(|layout| {
            let logical_rows = match layout {
                HistoryDomainLayout::Shared { .. } => 0,
                _ => layout
                    .geometry(&layouts, rows.context, rows.max_advance)
                    .map_err(|error| error.to_string())?
                    .reserved_rows(rows.rows(layout)?)
                    .ok_or("history rows overflow whole pages")?,
            };
            Ok(HistoryDomainPlan {
                logical_rows,
                layout: layout.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let domains = history
        .iter()
        .filter(|plan| !matches!(plan.layout, HistoryDomainLayout::Shared { .. }))
        .map(|plan| {
            let row_bytes = plan.layout.row_bytes().map_err(|error| error.to_string())?;
            let geometry = plan
                .layout
                .geometry(&layouts, rows.context, rows.max_advance)
                .map_err(|error| error.to_string())?;
            Ok(HistoryStorePlan {
                kind: plan.layout.kind(),
                components: plan.layout.components().to_vec(),
                rows: plan.logical_rows,
                row_bytes,
                slab_rows: u32::try_from(geometry.slab_rows)
                    .map_err(|_| "history slab rows exceed u32")?,
                page_rows: u32::try_from(geometry.page_rows)
                    .map_err(|_| "history page rows exceed u32")?,
                span_limit: geometry.span_limit,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let recurrent_bank_bytes = recurrent_bank_bytes(&recurrent_components)?;
    let recurrent_pool_bytes = recurrent_bank_bytes
        .checked_mul(
            u64::try_from(
                bank_capacity
                    .storage_total()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|_| "bank capacity exceeds u64")?,
        )
        .ok_or("recurrent pool byte count overflow")?;
    Ok(StateStorePlan {
        context_rows: rows.context,
        max_advance: rows.max_advance,
        history,
        domains,
        recurrent_components,
        bank_capacity,
        recurrent_bank_bytes,
        zero_seed_bytes: recurrent_bank_bytes,
        recurrent_pool_bytes,
    })
}

#[cfg(test)]
mod slab_plan_tests {
    use super::*;
    use magnitude_state::LayerRef;
    use seismic::{BackendName, DType, DeviceCatalog};

    fn placed_qwen_fixture() -> (ModelDefinition, ModelLoadPlan, ResourceLimits) {
        use crate::planning::tests::{qwen4exp_fixture_definition, stored_table_manifest};
        use crate::{ComponentSelection, ExecutionPath};
        let mut definition = qwen4exp_fixture_definition();
        // The shared fixture targets Metal/CPU with two value heads. CUDA's
        // recurrent gate requires NV % 4 == 0; retain every operator while
        // giving this CUDA projection a supported recurrent geometry.
        let magnitude_family_contracts::Operator::GatedDelta(delta) =
            &mut definition.decoder.blocks[0].sublayers[0].op else { panic!("recurrent fixture") };
        delta.value_heads = 4;
        let channels = (2 * delta.key_heads + delta.value_heads) * delta.width;
        delta.query_key_value.shape[0] = channels;
        delta.gate.shape[0] = delta.value_heads * delta.width;
        delta.alpha.shape[0] = delta.value_heads;
        delta.beta.shape[0] = delta.value_heads;
        delta.convolution.shape[0] = channels;
        delta.decay.shape[0] = delta.value_heads;
        delta.time_bias.shape[0] = delta.value_heads;
        delta.output.shape[1] = delta.value_heads * delta.width;
        definition.decoder.blocks.extend(definition.decoder.blocks.clone());
        let mut manifest = stored_table_manifest(&definition);
        // Exercise the registered packed expert conversion, as the deployed
        // model does. The base fixture's F16 experts cannot be losslessly
        // repacked into CUDA's BF16 resident dense representation.
        let mut offset = 0;
        for tensor in &mut manifest.target.tensors {
            if tensor.name.ends_with("_exps") {
                tensor.encoding = magnitude_artifacts::gguf::Encoding::Q4_0;
                tensor.nbytes = tensor.shape.iter().product::<u64>() / 32 * 18;
            }
            tensor.offset = offset;
            offset += tensor.nbytes;
        }
        manifest.target.files[0].size = offset;
        let limits = ResourceLimits {
            max_launch_rows: 16, max_launch_slots: 16, max_selected_rows: 16,
            max_drafting_slots: 16, exported_logits_rows: 0,
            max_images_per_request: 1, max_image_cells: 0, lookahead: false,
            group_capacity: NonZeroUsize::MIN,
        };
        let load = ModelLoadPlan::derive(&manifest, &definition,
            ComponentSelection { head: false, vision: false },
            crate::resident_layout(ExecutionPath::Native, BackendName::Cuda),
        ).unwrap().with_shared_expert_bank(9, 2).unwrap();
        (definition, load, limits)
    }

    #[test]
    fn placed_qwen_state_and_local_caches_certify_each_range() {
        use crate::{DecoderPlacement, DecoderRange};
        use seismic::DeviceSelector;
        let (definition, load, limits) = placed_qwen_fixture();
        let placement = DecoderPlacement::new(
            4,
            vec![
                DecoderRange {
                    device: DeviceSelector::Cuda { uuid: [1; 16] },
                    blocks: 0..2,
                },
                DecoderRange {
                    device: DeviceSelector::Cuda { uuid: [2; 16] },
                    blocks: 2..4,
                },
            ],
        )
        .unwrap();
        let capacities = [
            ResourceCapacity {
                domain_bytes: 64 << 20,
                tensor_operations: TensorOperations::Absent,
            },
            ResourceCapacity {
                domain_bytes: 128 << 20,
                tensor_operations: TensorOperations::Absent,
            },
        ];
        let state = ResourcePlanner::state_plan_placed(
            &definition,
            &load,
            PlannedMethod::Plain,
            KvCodec::Dense,
            limits,
            &placement,
            &capacities,
        )
        .unwrap();
        assert_eq!(state.placement(), Some(&placement));
        assert_eq!(state.history_owners(), [0, 1]);
        assert!(state.recurrent_owners().contains(&0));
        assert!(state.recurrent_owners().contains(&1));
        assert_eq!(
            state.capacity_bytes.domain_bytes,
            capacities[0].domain_bytes
        );
        let owner_row_bytes: u64 = state
            .target_state()
            .domains
            .iter()
            .zip(state.history_owners())
            .filter(|(_, owner)| **owner == 0)
            .map(|(domain, _)| domain.row_bytes)
            .sum();
        assert_eq!(
            state.capacity().history_rows as u64,
            capacities[0].domain_bytes / owner_row_bytes
        );
        let startup = state.startup_state_bytes_by_owner().unwrap();
        assert_eq!(startup.len(), 2);
        assert!(startup.iter().all(|&bytes| bytes > 0));
        assert_eq!(
            startup.iter().sum::<u64>(),
            state.startup_state_bytes().unwrap()
        );
        assert!(state.fit_state_bytes(128, 2).is_err());
        assert!(state.allocate_target_state_placed(&[]).is_err());
        for (owner, range) in placement.ranges().iter().enumerate() {
            // Independent device assessments can differ by one seed slot.
            // The state layout still describes the same model inventory.
            let source = if owner == 0 {
                load.clone()
            } else {
                load.resident_inventory()
                    .with_shared_expert_bank(10, 2)
                    .unwrap()
            };
            let range_load = placement.load_range(owner, &source, Some(&[32])).unwrap();
            assert_eq!(range_load.derives_from(&load), owner == 0);
            assert!(range_load.shares_model_inventory(&load));
            let mut foreign = load.clone();
            foreign.target[0].component.identity = magnitude_artifacts::ArtifactIdentity([99; 32]);
            assert!(!range_load.shares_model_inventory(&foreign));
            let mut changed = load.clone();
            changed.target[0].source_bytes += 1;
            assert!(!range_load.shares_model_inventory(&changed));
            let local = range_load.graph_load();
            let physical = range_load.physical_load();
            assert_eq!(range_load.decoder_blocks(), definition.decoder.blocks.len());
            assert_eq!(
                range_load.weight_bytes(),
                super::weight_bytes_by_component(physical).unwrap()[0]
            );
            assert!(physical.target().len() < local.target().len());
            assert!(
                range_load.import_peak_bytes()
                    >= physical
                        .target()
                        .iter()
                        .map(|weight| weight.source_bytes)
                        .max()
                        .unwrap()
            );
            for weight in physical.target() {
                use magnitude_family_contracts::WeightScope;
                match weight.role.scope {
                    WeightScope::Target => {}
                    WeightScope::TargetSublayer(index)
                    | WeightScope::TargetBranch {
                        sublayer: index, ..
                    } => assert!(range.blocks.contains(&(index.block as usize))),
                    _ => panic!("non-target weight in range inventory"),
                }
            }
            let bank = local.expert_bank().unwrap();
            assert_eq!(bank.layers.len(), 2);
            let programs = local.program_plan(&definition, KvCodec::Dense).unwrap();
            crate::programs::native_target_graph::checked_target_family_in_range(
                BackendName::Cuda,
                &local,
                &definition.decoder,
                &state,
                programs.target(),
                limits,
                range.blocks.clone(),
            )
            .unwrap();
            for weight in local.target() {
                if let Some(original) = load.expert_bank().unwrap().resident_plan(weight.role) {
                    if bank.holds(weight.role) {
                        assert_eq!(weight.source_bytes, 0);
                        assert_eq!(weight.shape[0], 32);
                    } else {
                        assert_eq!(weight, original);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires two CUDA devices"]
    fn placed_qwen_state_allocates_on_two_cuda_devices() {
        use crate::{DecoderPlacement, DecoderRange};
        let catalog = DeviceCatalog::discover().unwrap();
        let topology = catalog.topology();
        let devices = topology.devices().iter().filter(|device| device.backend == BackendName::Cuda)
            .take(2).map(|device| Rc::new(catalog.open(device.id).unwrap())).collect::<Vec<_>>();
        assert_eq!(devices.len(), 2);
        let (definition, load, limits) = placed_qwen_fixture();
        let placement = DecoderPlacement::new(4, devices.iter().enumerate().map(|(owner, device)| DecoderRange {
            device: device.info().selector, blocks: 2 * owner..2 * owner + 2,
        }).collect()).unwrap();
        let capacities = vec![ResourceCapacity { domain_bytes: 256 << 20,
            tensor_operations: TensorOperations::Absent }; 2];
        let state = ResourcePlanner::state_plan_placed(&definition, &load, PlannedMethod::Plain, KvCodec::Dense,
            limits, &placement, &capacities).unwrap();
        let baseline = devices.iter().map(|device| device.memory_usage().charged).collect::<Vec<_>>();
        assert!(state.allocate_target_state_placed(&[devices[1].clone(), devices[0].clone()]).is_err());
        let previous_limit = devices[1].memory_usage().limit;
        devices[1].set_memory_limit(Some(baseline[1]));
        let refused = state.allocate_target_state_placed(&devices);
        devices[1].set_memory_limit(previous_limit);
        assert!(refused.is_err());
        assert_eq!(devices.iter().map(|device| device.memory_usage().charged).collect::<Vec<_>>(), baseline);
        let bindings = state.allocate_target_state_placed(&devices).unwrap();
        let projected = state.startup_state_bytes_by_owner().unwrap();
        for (owner, device) in devices.iter().enumerate() {
            assert_eq!(bindings.committed_bytes_on(device), projected[owner]);
            assert_eq!(device.memory_usage().charged - baseline[owner], projected[owner]);
        }
        for (index, &owner) in state.recurrent_owners().iter().enumerate() {
            assert!(bindings.recurrent_device(index).unwrap().same_device(&devices[owner]));
        }
        for (index, &owner) in state.history_owners().iter().enumerate() {
            assert!(bindings.history_device(HistoryDomainId(index)).unwrap().same_device(&devices[owner]));
        }
        drop(bindings);
        assert_eq!(devices.iter().map(|device| device.memory_usage().charged).collect::<Vec<_>>(), baseline);
    }

    #[test]
    fn listed_history_classes_bound_capacity_for_small_and_large_extents() {
        for most in [1, 3, 5, 16, 17, 257, 1024] {
            let plan = HistoryStorePlan {
                kind: HistoryDomainKind::Token,
                components: Vec::new(),
                rows: most * magnitude_state::SLAB_ROW_TILE,
                row_bytes: 1,
                slab_rows: magnitude_state::SLAB_ROW_TILE as u32,
                page_rows: magnitude_state::SLAB_ROW_TILE as u32,
                span_limit: most,
            };
            // A form whose work follows the tiles a launch sees has one
            // class: one request's worth.
            assert_eq!(
                plan.listed_tile_classes(DecodedHistoryForm::Staged).unwrap(),
                [most as u64]
            );
            // A form that decodes into a window seals no class below its
            // floor: powers of two from it, then one request's worth.
            let windowed = plan
                .listed_tile_classes(DecodedHistoryForm::Windowed)
                .unwrap();
            assert_eq!(windowed.last(), Some(&(most as u64)));
            assert!(windowed.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(
                windowed[..windowed.len() - 1]
                    .iter()
                    .all(|tiles| *tiles >= 16 && tiles.is_power_of_two())
            );
        }
    }

    /// The form a backend's device has declares its listing classes: no
    /// graph is sealed for a class its form does nothing different at.
    #[test]
    fn the_decoded_history_form_declares_its_smallest_listing_class() {
        let form = |backend, tensor_operations, width| {
            DecodedHistoryForm::of(backend, tensor_operations, width)
        };
        assert_eq!(form(BackendName::Metal, true, 64), Some(DecodedHistoryForm::Windowed));
        assert_eq!(form(BackendName::Metal, false, 256), Some(DecodedHistoryForm::Windowed));
        assert_eq!(form(BackendName::Metal, false, 64), None);
        assert_eq!(form(BackendName::Cuda, false, 64), Some(DecodedHistoryForm::Staged));
        assert_eq!(form(BackendName::Cpu, false, 256), None);
        assert_eq!(DecodedHistoryForm::Windowed.smallest_listing_tiles(), Some(16));
        assert_eq!(DecodedHistoryForm::Staged.smallest_listing_tiles(), None);
    }

    #[test]
    fn startup_and_fit_charges_match_slab_storage() {
        let history = ComponentDescriptor::new(
            LayerRef::Target(0),
            KvCodec::Dense.spec(DType::BF16, 32, 32),
            1,
        )
        .unwrap();
        let plan = state_store_plan(
            HistoryRows {
                context: 600_000,
                max_advance: 512,
                token: 1_200_000,
                domain_bytes: 1 << 30,
            },
            vec![HistoryDomainLayout::Token {
                components: vec![history],
            }],
            vec![ComponentSpec {
                shape: vec![4 * 1024 * 1024],
                dtype: DType::F32,
            }],
            BankCapacity {
                active: 2,
                in_flight: 2,
                retained: 1,
            },
        )
        .unwrap();
        let history = plan.domains[0].slab_layout().unwrap();
        let banks = plan.bank_slab_layout().unwrap().unwrap();
        assert_eq!(plan.bank_slab_banks().unwrap(), 4);
        assert_eq!(
            plan.initial_committed_bytes().unwrap(),
            history.address_table_bytes
                + history.slab_bytes
                + banks.address_table_bytes
                + banks.slab_bytes
        );
        let device = Rc::new(
            DeviceCatalog::discover()
                .unwrap()
                .open_backend(BackendName::Cpu)
                .unwrap(),
        );
        let store = plan.allocate(device).unwrap();
        assert_eq!(
            store.committed_bytes(),
            plan.initial_committed_bytes().unwrap()
        );

        let history_depth = u64::from(plan.domains[0].slab_rows) + 1;
        assert_eq!(
            plan.history_bytes_at_depth(0).unwrap(),
            history.address_table_bytes + history.slab_bytes
        );
        assert_eq!(
            plan.history_bytes_at_depth(history_depth).unwrap(),
            history.address_table_bytes + 2 * history.slab_bytes
        );
        assert_eq!(
            plan.bank_bytes_at_count(5).unwrap(),
            banks.address_table_bytes + 2 * banks.slab_bytes
        );
    }
}

/// History bytes per token of context: the rows of every Token domain, and
/// a `rate`-th of every Block domain's (rounded up). Window domains hold a
/// bounded number of rows whatever the context (see
/// `StateStorePlan::history_bytes_at_depth`).
pub(super) fn history_row_bytes(domains: &[HistoryDomainLayout]) -> Result<u64, String> {
    domains
        .iter()
        .try_fold(0u64, |total, domain| {
            let row_bytes = domain.row_bytes().map_err(|error| error.to_string())?;
            // A Block domain keeps a row per `rate` tokens.
            let bytes = match domain.kind() {
                HistoryDomainKind::Token => row_bytes,
                HistoryDomainKind::Block { rate } => row_bytes.div_ceil(rate as u64),
                HistoryDomainKind::Window { .. } | HistoryDomainKind::Shared { .. } => 0,
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| "history row byte count overflow".into())
        })
}

pub(super) fn recurrent_bank_bytes(components: &[ComponentSpec]) -> Result<u64, String> {
    components.iter().try_fold(0u64, |total, component| {
        let bytes = u64::try_from(component.bytes()?)
            .map_err(|_| "recurrent component bytes exceed u64")?;
        total
            .checked_add(bytes)
            .ok_or_else(|| "recurrent bank byte count overflow".into())
    })
}
