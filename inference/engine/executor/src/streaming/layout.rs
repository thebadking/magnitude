//! The device tensors of a streamed load (spec 4.3): the step counter, every
//! routed layer's directory (slot map, stamps, request region, hint rings,
//! convert lists) and the staging pools, as the graphs' bank ports declare
//! them and as the load allocates them.

use crate::ModelLoadPlan;
use crate::planning::{BANK_KINDS, ExpertBankPlan};
use crate::programs::native_target_graph::BankPart;
use magnitude_family_contracts::{WeightKind, WeightRole, WeightScope};
use crate::operators::routed::fused_graph::DECODE_ROWS;
use seismic::{Device, Element, Tensor};

/// Prediction distances (spec 4.7): a routed layer hints the layers one and
/// two ahead.
pub(crate) const DISTANCES: [u8; 2] = [1, 2];

/// Words of a layer's request region: the request, acknowledgement,
/// failure and count words, then every choice of the decode form's rows
/// (`routed_route_streamed`).
pub(crate) fn control_words(selected: u64) -> u64 {
    4 + DECODE_ROWS * selected
}

/// Words of a hint ring: the sequence and count, then every choice of the
/// decode form's rows (`routed_route_streamed`).
pub(crate) fn ring_words(selected: u64) -> u64 {
    2 + DECODE_ROWS * selected
}

/// A dedicated device's host conveyor holds two scratch rounds ahead.
/// Its records are replenished as converts finish. Pinning every expert
/// of every source format would displace the host's model page cache.
pub(crate) fn host_conveyor_records(experts: u64, selected: u64) -> u64 {
    experts.min(2 * scratch_round_records(selected))
}

/// Staging records a decode step's hints may hold (spec 4.7): one row's
/// predicted experts for each distance ahead.
pub(crate) fn hint_records(selected: u64) -> u64 {
    DISTANCES.len() as u64 * selected
}

/// A staging pool's own records: two scratch-fill rounds, one filling while
/// the device converts the other, which hold the hint window beside every
/// demand read of the widest bank-served launch (the decode form's rows).
pub(crate) fn demand_records(selected: u64) -> u64 {
    hint_records(selected) + DECODE_ROWS * selected
}

/// Records a scratch fill stages per round (spec 4.9). Both rounds must
/// leave the hint window free: a layer-major group can mix scratch-served
/// chunks with small chunks that publish decode hints.
pub(crate) fn scratch_round_records(selected: u64) -> u64 {
    ((demand_records(selected) - hint_records(selected)) / 2).max(1)
}

/// Rounds a scratch-served layer's graph holds: enough to deliver every
/// expert.
pub(crate) fn scratch_rounds(experts: u64, selected: u64) -> u64 {
    experts.div_ceil(scratch_round_records(selected))
}

/// Words of a load's scratch request region (`routed_scratch_request`):
/// the request, layer, failure and a reserved word, then each round's
/// acknowledgement and each round's arrival (`routed_scratch_wait`).
pub(crate) fn scratch_control_words(rounds: u64) -> u64 {
    4 + 2 * rounds
}

fn bank(load: &ModelLoadPlan) -> Result<&ExpertBankPlan, String> {
    load.expert_bank()
        .ok_or_else(|| "a bank port of a load that does not stream".to_owned())
}

fn layer_tensor(
    bank: &ExpertBankPlan,
    scope: WeightScope,
    kind: u8,
) -> Result<crate::planning::BankTensor, String> {
    let layer = bank
        .layers
        .iter()
        .find(|layer| layer.scope == scope)
        .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
    layer
        .tensors
        .get(usize::from(kind))
        .copied()
        .ok_or_else(|| format!("routed tensor kind {kind} of {BANK_KINDS:?}"))
}

/// The element and extents of `part` of the routed layer of `scope`.
pub(crate) fn bank_port(
    load: &ModelLoadPlan,
    scope: WeightScope,
    part: BankPart,
) -> Result<(Element, Vec<u64>), String> {
    let bank = bank(load)?;
    let words = |count: u64| Ok((Element::i32(), vec![count]));
    match part {
        BankPart::Step | BankPart::Heartbeat => words(1),
        BankPart::SlotMap => words(bank.experts),
        BankPart::Stamps => words(bank.records(scope)?),
        BankPart::Commits => words(bank.records(scope)?),
        BankPart::Control => words(bank.control_words()),
        BankPart::Placement => Ok((Element::i32(), vec![5, bank.placement_choices()])),
        BankPart::Ring { .. } => words(bank.ring_words()),
        BankPart::Staging { kind } => {
            let pool = &bank.pools[layer_tensor(bank, scope, kind)?.pool];
            Ok((
                pool.source,
                vec![bank.graph_staging_records(), pool.matrix[0], pool.matrix[1]],
            ))
        }
        BankPart::Conveyor { kind } => {
            let pool = &bank.pools[layer_tensor(bank, scope, kind)?.pool];
            Ok((
                pool.source,
                vec![bank.conveyor_records(), pool.matrix[0], pool.matrix[1]],
            ))
        }
        BankPart::Scratch { kind } => {
            let scratch = &bank.scratch[layer_tensor(bank, scope, kind)?.scratch];
            Ok((
                scratch.element,
                vec![bank.experts, scratch.matrix[0], scratch.matrix[1]],
            ))
        }
        BankPart::ScratchControl => words(bank.scratch_control_words()),
        BankPart::ScratchFlags => words(bank.experts),
        BankPart::ScratchList { .. } => words(bank.list_words()),
        BankPart::CopyList { .. } | BankPart::SeedList => words(bank.copy_list_words()),
    }
}

/// What a streamed block's graph depends on beyond its own weights: its
/// position at the start of the bank, and each layer it hints
/// (spec 4.7) with its distance in blocks and the planned norm and router it
/// predicts through. Blocks that agree share a graph.
pub(crate) fn block_key(load: &ModelLoadPlan, block: usize) -> Result<Option<String>, String> {
    let Some(bank) = load.expert_bank() else {
        return Ok(None);
    };
    let block_of = |scope: WeightScope| match scope {
        WeightScope::TargetSublayer(index) => Ok(index.block as usize),
        _ => Err(format!("streamed layer {scope:?} outside the target")),
    };
    let mut position = None;
    for (index, layer) in bank.layers.iter().enumerate() {
        if block_of(layer.scope)? == block {
            position = Some(index);
        }
    }
    let Some(position) = position else {
        return Ok(None);
    };
    let mut ahead = Vec::new();
    for distance in DISTANCES {
        let Some(layer) = bank.layers.get(position + usize::from(distance)) else {
            break;
        };
        let planned = |kind| {
            load.weights()
                .find(|plan| plan.role == WeightRole { scope: layer.scope, kind })
                .map(|plan| (plan.resident, plan.shape.clone()))
                .ok_or_else(|| format!("{:?} of {:?} is not planned", kind, layer.scope))
        };
        // InputNorm is absent for HyperConnectionMix layers (NM=0); the
        // kernel uses sum_parts instead of rms_norm for those.
        let norm_info = load.weights()
            .find(|plan| plan.role == WeightRole {
                scope: layer.scope,
                kind: WeightKind::InputNorm,
            })
            .map(|plan| (plan.resident, plan.shape.clone()));
        ahead.push((
            block_of(layer.scope)? - block,
            norm_info,
            planned(WeightKind::Router)?,
            planned(WeightKind::SharedRouter)?,
        ));
    }
    Ok(Some(format!("{:?}", (position == 0, ahead))))
}

/// One routed layer's directory tensors.
pub(crate) struct LayerTensors {
    pub scope: WeightScope,
    pub slot_map: Tensor,
    /// The layer's bank group's stamps and commits, shared by its layers.
    pub stamps: Tensor,
    pub commits: Tensor,
    pub control: Tensor,
    /// [5, P]: per choice, the slot, the staging records and the
    /// acknowledgement of an absent expert (`routed_bank_fill`).
    pub placement: Tensor,
    /// This layer's hints for the layers one and two ahead.
    pub rings: [Tensor; 2],
    /// The layer's scratch request region.
    pub scratch_control: Tensor,
    pub pools: [usize; 3],
    pub scratch: [usize; 3],
}

/// A streamed load's device tensors, standalone allocations the host
/// addresses while steps run (`Tensor::host_window`).
pub(crate) struct BankTensors {
    pub step: Tensor,
    device_step: Option<Tensor>,
    pub heartbeat: Tensor,
    pub layers: Vec<LayerTensors>,
    /// Each staging pool's own records.
    pub pools: Vec<Tensor>,
    /// CUDA source views backed by three device arenas; the host pools
    /// remain the file workers' destinations.
    pub device_pools: Option<Vec<Tensor>>,
    device_conveyors: Vec<Tensor>,
    /// Each staging pool's conveyor (spec 4.9): a view of its routed
    /// tensor's conveyor arena.
    pub conveyors: Vec<Tensor>,
    /// The scratch fill's: one bank per resident form, the request region,
    /// each round's convert lists, the copy lists and the seed list (spec 4.9).
    pub scratch: Vec<Tensor>,
    pub scratch_flags: Tensor,
    pub scratch_lists: Vec<[Tensor; 3]>,
    pub copy_lists: [Tensor; 3],
    pub seed_list: Tensor,
    pub device_scratch_lists: Vec<[Tensor; 3]>,
    pub device_copy_lists: Option<[Tensor; 3]>,
    pub device_seed_list: Option<Tensor>,
}

impl BankTensors {
    /// Allocate the directory of a streamed load over its shared bank plan.
    /// `lender` is the byte arenas of the bank group lending the conveyor its
    /// records (`ConveyorPlan::lender`); without one, the conveyor's arenas
    /// are allocated here. All fallible allocations remain private until this
    /// complete value returns; the construction peak therefore equals its
    /// persistent device charge.
    pub(crate) fn allocate(
        device: &Device,
        bank: &ExpertBankPlan,
        shared: &super::shared_bank::SharedBankPlan,
        lender: Option<[Tensor; 3]>,
        checkpoint: &mut dyn FnMut() -> Result<(), crate::ResidencyError>,
    ) -> Result<Self, crate::ResidencyError> {
        if shared != &super::shared_bank::SharedBankPlan::admitted(bank)? {
            return Err("shared directory plan differs from bank metadata".into());
        }
        let words = |count: u64| {
            Tensor::shared(device, Element::i32(), &[count])
                .map_err(|error| format!("bank directory allocation failed: {error}"))
        };
        let group_words = shared
            .groups
            .iter()
            .map(|group| Ok((words(group.records)?, words(group.records)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let layers = bank
            .layers
            .iter()
            .enumerate()
            .map(|(position, layer)| {
                let (stamps, commits) = group_words[shared.group_of_layer[position]].clone();
                Ok(LayerTensors {
                    scope: layer.scope,
                    slot_map: words(bank.experts)?,
                    stamps,
                    commits,
                    control: words(bank.control_words())?,
                    placement: Tensor::shared(
                        device,
                        Element::i32(),
                        &[5, bank.placement_choices()],
                    )
                    .map_err(|error| format!("bank directory allocation failed: {error}"))?,
                    rings: [words(bank.ring_words())?, words(bank.ring_words())?],
                    scratch_control: words(bank.scratch_control_words())?,
                    pools: layer.tensors.map(|tensor| tensor.pool),
                    scratch: layer.tensors.map(|tensor| tensor.scratch),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let pools = bank
            .pools
            .iter()
            .map(|pool| {
                Tensor::shared(
                    device,
                    pool.source,
                    &[bank.staging_records, pool.matrix[0], pool.matrix[1]],
                )
                .map_err(|error| format!("staging allocation failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let carriers = match (bank.conveyor.lender, lender) {
            (Some(_), Some(arenas)) => arenas,
            (None, None) => {
                let arena = |kind| {
                    Tensor::shared(device, Element::bool(), &[bank.conveyor_arena_bytes(kind)?])
                        .map_err(|error| format!("conveyor arena allocation failed: {error}"))
                };
                [arena(0)?, arena(1)?, arena(2)?]
            }
            _ => return Err("the conveyor's lender differs from its plan".into()),
        };
        let conveyors = bank
            .pools
            .iter()
            .zip(&bank.conveyor.offsets)
            .map(|(pool, &offset)| {
                carriers[pool.kind]
                    .region(
                        pool.source,
                        &[bank.conveyor_records(), pool.matrix[0], pool.matrix[1]],
                        offset,
                    )
                    .map_err(|error| format!("conveyor view failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let device_pools = if bank.device_staging {
            let arenas = (0..3).map(|kind| {
                Tensor::zeros(device, Element::bool(), &[bank.device_staging_arena_bytes(kind)?])
                    .map_err(|error| format!("device source arena allocation failed: {error}"))
            }).collect::<Result<Vec<_>, String>>()?;
            Some(bank.pools.iter().map(|pool| {
                arenas[pool.kind].region(pool.source,
                    &[bank.graph_staging_records(), pool.matrix[0], pool.matrix[1]], 0)
                    .map_err(|error| format!("device source view failed: {error}"))
            }).collect::<Result<Vec<_>, String>>()?)
        } else { None };
        let device_conveyors = device_pools.as_ref().map(|pools| pools.iter().map(|pool| {
            pool.slice_leading(0, bank.conveyor_records())
                .map_err(|error| format!("device conveyor view failed: {error}"))
        }).collect::<Result<Vec<_>, String>>()).transpose()?.unwrap_or_default();
        let lists = || -> Result<[Tensor; 3], String> {
            Ok([
                words(bank.list_words())?,
                words(bank.list_words())?,
                words(bank.list_words())?,
            ])
        };
        let scratch_lists = (0..bank.rounds())
            .map(|_| lists())
            .collect::<Result<Vec<_>, String>>()?;
        let copy_words = bank.copy_list_words();
        let step = words(1)?;
        let heartbeat = words(1)?;
        let scratch_flags = words(bank.experts)?;
        let copy_lists = [words(copy_words)?, words(copy_words)?, words(copy_words)?];
        let seed_list = words(copy_words)?;
        let device_words = |count| Tensor::zeros(device, Element::i32(), &[count])
            .map_err(|error| format!("device source list allocation failed: {error}"));
        let device_step = bank.device_staging.then(|| device_words(1)).transpose()?;
        let device_scratch_lists = if bank.device_staging {
            (0..bank.rounds()).map(|_| Ok([
                device_words(bank.list_words())?, device_words(bank.list_words())?,
                device_words(bank.list_words())?,
            ])).collect::<Result<Vec<_>, String>>()?
        } else { Vec::new() };
        let device_copy_lists = if bank.device_staging { Some([
            device_words(copy_words)?, device_words(copy_words)?, device_words(copy_words)?,
        ]) } else { None };
        let device_seed_list = bank.device_staging.then(|| device_words(copy_words)).transpose()?;
        checkpoint()?;
        // One arena per routed tensor, sized for its largest form; each
        // form is a view of it at offset 0 (one layer fills it at a time).
        let arenas = (0..3)
            .map(|kind| {
                Tensor::zeros(device, Element::bool(), &[bank.scratch_arena_bytes(kind)?])
                    .map_err(|error| format!("scratch arena allocation failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let scratch = bank
            .scratch
            .iter()
            .map(|scratch| {
                arenas[scratch.kind]
                    .region(
                        scratch.element,
                        &[bank.experts, scratch.matrix[0], scratch.matrix[1]],
                        0,
                    )
                    .map_err(|error| format!("scratch bank view failed: {error}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            step,
            device_step,
            heartbeat,
            layers,
            pools,
            device_pools,
            device_conveyors,
            conveyors,
            scratch,
            scratch_flags,
            scratch_lists,
            copy_lists,
            seed_list,
            device_scratch_lists,
            device_copy_lists,
            device_seed_list,
        })
    }

    /// The tensor `part` of the routed layer of `scope` binds.
    pub(crate) fn part(&self, scope: WeightScope, part: BankPart) -> Result<&Tensor, String> {
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.scope == scope)
            .ok_or_else(|| format!("{scope:?} is not a streamed routed layer"))?;
        let index = |value: u8, bound: usize| {
            let value = usize::from(value);
            (value < bound)
                .then_some(value)
                .ok_or_else(|| format!("bank part {part:?} out of range"))
        };
        Ok(match part {
            BankPart::Step => self.device_step.as_ref().unwrap_or(&self.step),
            BankPart::Heartbeat => &self.heartbeat,
            BankPart::SlotMap => &layer.slot_map,
            BankPart::Stamps => &layer.stamps,
            BankPart::Commits => &layer.commits,
            BankPart::Control => &layer.control,
            BankPart::Placement => &layer.placement,
            BankPart::Ring { distance } => {
                &layer.rings[index(distance.wrapping_sub(1), 2)?]
            }
            BankPart::Staging { kind } => &self.device_pools.as_ref().unwrap_or(&self.pools)
                [layer.pools[index(kind, 3)?]],
            BankPart::Conveyor { kind } => {
                let conveyors = if self.device_pools.is_some() { &self.device_conveyors }
                    else { &self.conveyors };
                &conveyors[layer.pools[index(kind, 3)?]]
            }
            BankPart::Scratch { kind } => &self.scratch[layer.scratch[index(kind, 3)?]],
            BankPart::ScratchControl => &layer.scratch_control,
            BankPart::ScratchFlags => &self.scratch_flags,
            BankPart::ScratchList { round, kind } => {
                let lists = if self.device_pools.is_some() { &self.device_scratch_lists }
                    else { &self.scratch_lists };
                &lists[index(round, lists.len())?][index(kind, 3)?]
            }
            BankPart::CopyList { kind } => &self.device_copy_lists.as_ref()
                .unwrap_or(&self.copy_lists)[index(kind, 3)?],
            BankPart::SeedList => self.device_seed_list.as_ref().unwrap_or(&self.seed_list),
        })
    }
}
