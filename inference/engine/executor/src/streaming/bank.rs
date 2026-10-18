//! A streamed target's banks as a load holds them: the device tensors the
//! graphs bind and the watcher that serves them (spec 4.3–4.7).

use super::layout::BankTensors;
use super::runtime::{
    Heartbeat, Loan, ScratchFill, StagingPool, StoredTensor, StreamedLayer, StreamingRuntime,
};
use super::source::{ExpertSource, ModelFile};
use super::words::{SharedBytes, SharedWords};
use crate::ResidentRoles;
use crate::planning::{BANK_KINDS, ExpertBankPlan};
use crate::residency::{ImportArtifactTensor, Stored};
use magnitude_artifacts::gguf::GgufArtifact;
use magnitude_family_contracts::{WeightRole, WeightScope};
use seismic::Device;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

/// The resident side of a streamed target: every bank starts empty (rule 5).
pub(crate) struct ResidentBank {
    pub host_experts: Option<Arc<crate::HostExpertCache>>,
    pub tensors: BankTensors,
    /// Admitted source/control storage actually held by this opened bank.
    pub source_control_bytes: u64,
    runtime: StreamingRuntime,
    /// The sequence of the last step submitted (4.3).
    submitted: AtomicI32,
}

/// One submitted bank-served step. Completing it reports the watcher's
/// failure of any of its requests (rule 6); a step dropped unfinished is
/// still counted complete, so the watcher stops polling for it.
pub(crate) struct BankStep {
    bank: Arc<ResidentBank>,
    sequence: i32,
    completed: bool,
}

impl BankStep {
    /// The step's sequence, which its streamed layers set the device's step
    /// to (`routed_step`).
    pub(crate) fn sequence(&self) -> i32 {
        self.sequence
    }

    pub(crate) fn complete(mut self) -> Result<(), String> {
        self.completed = true;
        self.bank.runtime.step_completed(self.sequence, true)
    }
}

impl Drop for BankStep {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.bank.runtime.step_completed(self.sequence, false);
        }
    }
}

impl std::fmt::Debug for BankStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BankStep")
            .field("sequence", &self.sequence)
            .finish()
    }
}

impl ResidentBank {
    /// Allocate the banks' tensors and start the watcher over `target`'s
    /// stored expert records. `sublayers` holds the empty bank weights,
    /// whose descriptors name the stored tensors; `lender` the byte arenas of
    /// the bank group the conveyor borrows records of (`ConveyorPlan`).
    pub(crate) fn open(
        device: &Device,
        plan: &ExpertBankPlan,
        target: &GgufArtifact,
        sublayers: &ResidentRoles,
        lender: Option<[seismic::Tensor; 3]>,
        host_experts: Option<Arc<crate::HostExpertCache>>,
        checkpoint: &mut dyn FnMut() -> Result<(), crate::ResidencyError>,
    ) -> Result<Self, crate::ResidencyError> {
        let shared = plan
            .shared
            .as_ref()
            .ok_or_else(|| "a streamed load without its shared bank plan".to_owned())?;
        let tensors = BankTensors::allocate(device, plan, shared, lender, checkpoint)?;
        let directory = super::shared_directory::SharedDirectory::new(shared, plan.experts)
            .map_err(|e| format!("shared bank directory: {e:?}"))?;
        let mut paths = Vec::new();
        let mut layers = Vec::with_capacity(plan.layers.len());
        for (layer, bank) in plan.layers.iter().zip(&tensors.layers) {
            let mut stored = [StoredTensor {
                file: 0,
                base: 0,
                bytes: 0,
                pool: 0,
            }; 3];
            for (kind, planned) in layer.tensors.iter().enumerate() {
                let role = WeightRole {
                    scope: layer.scope,
                    kind: BANK_KINDS[kind],
                };
                let descriptor = sublayers.get(role)?.descriptor();
                let source = ImportArtifactTensor::from_gguf(target, descriptor)
                    .map_err(|error| error.to_string())?;
                let Stored::GgmlBlocks { .. } = source.stored() else {
                    return Err(format!("{role:?} is not stored as packed records").into());
                };
                let (file, offset, bytes) = source.stored().file_range();
                if Some(bytes) != planned.stored_bytes.checked_mul(plan.experts) {
                    return Err(format!(
                        "{role:?} stores {bytes} bytes, not {} experts of {}",
                        plan.experts, planned.stored_bytes
                    )
                    .into());
                }
                if let Some(cache) = &host_experts {
                    cache.register_tensor(file.path(), offset, planned.stored_bytes, plan.experts, layer.scope)?;
                }
                let file = match paths.iter().position(|path| path == file.path()) {
                    Some(index) => index,
                    None => {
                        paths.push(file.path().to_owned());
                        paths.len() - 1
                    }
                };
                stored[kind] = StoredTensor {
                    file,
                    base: offset,
                    bytes: planned.stored_bytes,
                    pool: planned.pool,
                };
            }
            let words = |tensor: &seismic::Tensor| SharedWords::new(tensor.clone());
            layers.push(StreamedLayer::new(
                words(&bank.slot_map)?,
                words(&bank.stamps)?,
                words(&bank.commits)?,
                words(&bank.control)?,
                words(&bank.placement)?,
                words(&bank.scratch_control)?,
                [words(&bank.rings[0])?, words(&bank.rings[1])?],
                stored,
            ));
        }
        SharedWords::new(tensors.step.clone())?.fill(0);
        let records =
            u32::try_from(plan.staging_records).map_err(|_| "staging records exceed u32")?;
        let hints = u32::try_from(crate::streaming::layout::hint_records(plan.selected))
            .map_err(|_| "hint records exceed u32")?;
        let conveyor = u32::try_from(plan.conveyor_records())
            .map_err(|_| "conveyor records exceed u32")?;
        let pools = plan
            .pools
            .iter()
            .zip(tensors.pools.iter().zip(&tensors.conveyors))
            .enumerate()
            .map(|(index, (pool, (own, carried)))| {
                let staging = StagingPool::new(
                    SharedBytes::new(own.clone())?,
                    SharedBytes::new(carried.clone())?,
                    records,
                    hints,
                    conveyor,
                    pool.record_bytes,
                );
                Ok(match &tensors.device_pools {
                    Some(device) => staging.with_device(device[index].clone()),
                    None => staging,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // The bank records the conveyor lies on, when a group lends them.
        let loan = match plan.conveyor.lender {
            Some(group) => {
                let end = usize::try_from(shared.groups[group].records)
                    .map_err(|_| "bank records exceed the host")?;
                let lent = usize::try_from(plan.conveyor.lent)
                    .map_err(|_| "lent records exceed the host")?;
                Some(Loan::new(group, end - lent..end))
            }
            None => None,
        };
        if paths.is_empty() {
            return Err("a streamed load without routed layers".into());
        }
        let files = paths
            .iter()
            .map(|path| ModelFile::open(path).map(|file| match &host_experts {
                Some(cache) => file.with_cache(path, cache.clone()),
                None => file,
            }))
            .collect::<Result<Vec<_>, _>>()?;
        let source = ExpertSource::new(files)?;
        let words = |tensor: &seismic::Tensor| SharedWords::new(tensor.clone());
        let list_words = |tensor: &seismic::Tensor, mirror: Option<&seismic::Tensor>| {
            let host = words(tensor)?;
            match mirror { Some(mirror) => host.with_mirror(mirror.clone()), None => Ok(host) }
        };
        let scratch = ScratchFill::new(
            words(&tensors.scratch_flags)?,
            tensors
                .scratch_lists
                .iter()
                .enumerate()
                .map(|(round, [gate, up, down])| {
                    let mirror = tensors.device_scratch_lists.get(round);
                    Ok([
                        list_words(gate, mirror.map(|lists| &lists[0]))?,
                        list_words(up, mirror.map(|lists| &lists[1]))?,
                        list_words(down, mirror.map(|lists| &lists[2]))?,
                    ])
                })
                .collect::<Result<Vec<_>, String>>()?,
            {
                let [gate, up, down] = &tensors.copy_lists;
                let mirror = tensors.device_copy_lists.as_ref();
                [
                    list_words(gate, mirror.map(|lists| &lists[0]))?,
                    list_words(up, mirror.map(|lists| &lists[1]))?,
                    list_words(down, mirror.map(|lists| &lists[2]))?,
                ]
            },
            list_words(&tensors.seed_list, tensors.device_seed_list.as_ref())?,
            usize::try_from(plan.experts).map_err(|_| "experts exceed the host")?,
            usize::try_from(plan.round_records()).map_err(|_| "round records exceed the host")?,
        );
        let heartbeat = Heartbeat::new(words(&tensors.heartbeat)?);
        let runtime =
            StreamingRuntime::new(layers, scratch, heartbeat, pools, source, directory, loan)?;
        Ok(Self {
            host_experts,
            tensors,
            source_control_bytes: plan.shared_bytes()?.checked_add(plan.device_staging_bytes()?)
                .ok_or("bank source/control bytes overflow")?,
            runtime,
            submitted: AtomicI32::new(0),
        })
    }


    /// `count` steps are being submitted as one chain, bank-served or
    /// scratch-served (spec 4.5): one step, or a layer-major group, which
    /// interleaves their streamed layers on the device. Their sequences are
    /// the next `count`, in chain order. `seed_member` identifies each
    /// layer's last scratch-served member, before any small decode tail.
    pub(crate) fn begin_group(
        self: &Arc<Self>,
        count: usize,
        seed_member: impl Fn(WeightScope) -> Option<usize>,
    ) -> Vec<BankStep> {
        let seeds = self
            .tensors
            .layers
            .iter()
            .map(|layer| {
                seed_member(layer.scope).map(|index| {
                    assert!(index < count, "a seed member belongs to its group");
                    index
                })
            })
            .collect::<Vec<_>>();
        let count = i32::try_from(count).expect("a layer-major group's steps fit i32");
        let first = self.submitted.fetch_add(count, Ordering::AcqRel) + 1;
        self.runtime.steps_submitted(
            first,
            count,
            seeds.into_iter().map(|index| index.map(|index| first + index as i32)),
        );
        (first..first + count)
            .map(|sequence| BankStep {
                bank: Arc::clone(self),
                sequence,
                completed: false,
            })
            .collect()
    }
}
