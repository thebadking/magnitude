//! Decoder ownership independent of device allocation and graph preparation.
use magnitude_state::{HistoryDomainLayout, LayerRef, ModelStateLayout};
use seismic::DeviceSelector;
use std::ops::Range;

/// One contiguous range of global decoder blocks on one physical device.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecoderRange {
    pub device: DeviceSelector,
    pub blocks: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderPlacement {
    ranges: Vec<DecoderRange>,
}

/// Domain owners index the placement's ranges. Recurrent owners retain the
/// original component order, including n-gram banks following their mixer.
pub struct PlacedStateLayout {
    pub layout: ModelStateLayout,
    pub history_owners: Vec<usize>,
    pub recurrent_owners: Vec<usize>,
}

/// Graph metadata and physical holdings of one device. Outside block metadata
/// is retained only for deriving global topology, never for physical import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderRangeLoadPlan {
    range: DecoderRange,
    decoder_blocks: usize,
    source_load: super::ModelLoadPlan,
    graph_load: super::ModelLoadPlan,
    physical_load: super::ModelLoadPlan,
    weight_bytes: u64,
    import_peak_bytes: u64,
}

impl DecoderRangeLoadPlan {
    pub fn source_load(&self) -> &super::ModelLoadPlan { &self.source_load }

    pub(super) fn derives_from(&self, load: &super::ModelLoadPlan) -> bool {
        &self.source_load == load
    }

    /// Distributed state is common to the model, while each owner's cache
    /// capacity and host staging were admitted against that owner's device.
    pub(super) fn shares_model_inventory(&self, load: &super::ModelLoadPlan) -> bool {
        self.derives_from(load)
            || self.source_load.resident_inventory() == load.resident_inventory()
    }

    pub fn range(&self) -> &DecoderRange {
        &self.range
    }
    pub fn decoder_blocks(&self) -> usize {
        self.decoder_blocks
    }
    pub fn graph_load(&self) -> &super::ModelLoadPlan {
        &self.graph_load
    }
    pub fn physical_load(&self) -> &super::ModelLoadPlan {
        &self.physical_load
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn import_peak_bytes(&self) -> u64 {
        self.import_peak_bytes
    }

    /// Preparation finishes before graph pools or physical model weights
    /// exist. Use the graph inventory the factory actually prepares, which
    /// retains outside-block metadata, rather than the import inventory.
    pub fn preparation_peak_bytes(
        &self,
        definition: &magnitude_family_contracts::ModelDefinition,
        codec: magnitude_state::KvCodec,
        backend: seismic::BackendName,
        limits: super::ResourceLimits,
        tensor_operations: bool,
    ) -> Result<u64, String> {
        let programs = self.graph_load.program_plan(definition, codec)
            .map_err(|error| error.to_string())?;
        let invocation = crate::AttestedPrograms::planned_invocation_workspace_bytes(
            &programs,
            backend,
            limits,
            tensor_operations,
        )
        .map_err(|error| error.to_string())?;
        invocation
            .checked_add(crate::AttestedPrograms::qualification_peak_bytes(&self.graph_load)?)
            .and_then(|bytes| bytes.checked_add(seismic::recorded_graph_device_bytes(backend)))
            .ok_or_else(|| "range preparation peak overflows".into())
    }
}

impl DecoderPlacement {
    /// Select the largest local cache whose complete owner allocation fits.
    /// State and graph bounds use the same placement and physical inventory as
    /// final admission; another owner's capacity cannot fund this cache.
    pub fn load_range_with_cache_budget(
        &self,
        owner: usize,
        definition: &magnitude_family_contracts::ModelDefinition,
        draft: &super::ExecutionPlanDraft,
        capacities: &[super::ResourceCapacity],
        weight_cap: Option<u64>,
    ) -> Result<DecoderRangeLoadPlan, String> {
        let load = draft.load();
        let shared_in_host = draft.device().shared_in_host();
        let capacity = capacities
            .get(owner)
            .ok_or("unknown range capacity owner")?
            .domain_bytes;
        if self.ranges.get(owner).map(|range| range.device) != Some(draft.device().selector())
            || capacity != draft.device().assessment_capacity_bytes()
        {
            return Err("range cache capacity differs from the selected owner".into());
        }
        let initial = self.load_range(owner, load, None)?;
        let Some(bank) = initial.physical_load.expert_bank() else {
            return Ok(initial);
        };
        let Some(shared) = bank.shared.as_ref() else {
            return Ok(initial);
        };
        let device_weights = |load: &super::ModelLoadPlan| -> Result<u64, String> {
            let bytes = super::weight_bytes_by_component(load)?[0];
            let host = if shared_in_host {
                load.shared_weight_bytes()?
            } else {
                0
            };
            bytes
                .checked_sub(host)
                .ok_or_else(|| "host weights exceed weight charge".into())
        };
        let source_budget = device_weights(load)?;
        let state = super::ResourcePlanner::state_plan_placed(
            definition,
            load,
            draft.policy().method(),
            draft.policy().codec(),
            draft.policy().limits(),
            self,
            capacities,
        )?;
        let minimum = crate::operators::routed::fused_graph::DECODE_ROWS
            .checked_mul(bank.selected)
            .ok_or("route union overflow")?
            .min(bank.experts);
        // Evaluate exact inventories, including directory and staging changes.
        // Do not extrapolate physical charge from a per-record payload slope.
        for slots in (bank.slots..=bank.experts).rev() {
            let records = shared
                .groups
                .iter()
                .map(|group| {
                    slots
                        .checked_mul(group.layers.len() as u64)
                        .map(|records| records.max(minimum))
                        .ok_or("local expert record count overflow")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let candidate = self.load_range(owner, load, Some(&records))?;
            if device_weights(candidate.physical_load())? > capacity
                || weight_cap.is_some_and(|cap| candidate.weight_bytes() > cap)
            {
                continue;
            }
            let charge = super::assessment::range_memory_charge(
                definition, draft, &candidate, &state, owner,
            )?;
            let allocation = charge.device_allocation_bytes(shared_in_host)?;
            if allocation <= capacity {
                let memory = &charge.memory;
                let source = candidate.physical_load().target().iter()
                    .max_by_key(|weight| weight.source_bytes);
                eprintln!(
                    "magnitude-engine: range cache assessment {}",
                    serde_json::json!({
                        "blocks":candidate.range.blocks,"slots_per_layer":slots,
                        "physical_weight_bytes":candidate.weight_bytes(),"source_device_weight_budget":source_budget,
                        "device_allocation_bound_bytes":allocation,"device_capacity_bytes":capacity,
                        "prepared_resource_bound_bytes":memory.bounds.prepared_resource_bytes,
                        "startup_additional_bound_bytes":memory.bounds.startup_additional_bytes,
                        "preparation_allocation_bound_bytes":charge.preparation_bytes,
                        "qualification_peak_bytes":crate::AttestedPrograms::qualification_peak_bytes(candidate.graph_load())?,
                        "largest_import_source":source.map(|weight|serde_json::json!({
                            "name":weight.descriptor.name,"source_bytes":weight.source_bytes,
                            "resident_bytes":weight.resident_bytes,"shape":weight.shape,
                        })),
                        "fit_state_bytes":memory.exact_resident_bytes
                            .checked_sub(super::weight_bytes_by_component(candidate.physical_load())?.into_iter().sum())
                            .ok_or("range resident charge differs from weights")?,
                        "host_shared_weight_bytes":memory.shared_weight_bytes,
                        "host_shared_graph_upload_bytes":memory.bounds.shared_upload_bytes,
                        "fit_depth":charge.state_depth,"weight_cap":weight_cap,
                    })
                );
                return Ok(candidate);
            }
        }
        Err(format!(
            "no local expert cache fits owner {owner}'s complete allocation at capacity {capacity}, weight cap {weight_cap:?}"
        ))
    }

    pub fn new(blocks: usize, ranges: Vec<DecoderRange>) -> Result<Self, String> {
        if blocks == 0 || ranges.is_empty() {
            return Err("decoder placement needs blocks and devices".into());
        }
        let mut end = 0;
        for (index, range) in ranges.iter().enumerate() {
            if range.blocks.start != end || range.blocks.end <= end || range.blocks.end > blocks {
                return Err("decoder ranges must cover every block once, in order".into());
            }
            if ranges[..index]
                .iter()
                .any(|other| other.device == range.device)
            {
                return Err("a device must own one contiguous decoder range".into());
            }
            end = range.blocks.end;
        }
        if end != blocks {
            return Err("decoder placement does not cover the final block".into());
        }
        Ok(Self { ranges })
    }

    /// Exact weight/cache inventory for one owner. Entry and readout weights
    /// are currently duplicated and charged on each device, matching the range
    /// graph binder's entry/readout contract. All block weights stay local.
    pub fn load_range(
        &self,
        owner: usize,
        load: &super::ModelLoadPlan,
        records: Option<&[u64]>,
    ) -> Result<DecoderRangeLoadPlan, String> {
        use magnitude_family_contracts::WeightScope;
        let range = self
            .ranges
            .get(owner)
            .ok_or("unknown decoder range owner")?
            .clone();
        if load.vision.is_some() {
            return Err(
                "decoder range import requires text execution".into(),
            );
        }
        let mut graph_load = load.for_decoder_graph_range(range.blocks.clone(), records)?;
        // Only the entry owner imports the head and shares its embedding and
        // projection. Every target owner retains the conditioning annotation:
        // the final owner's readout must publish the original residual streams.
        if owner != 0 { graph_load.head = None; }
        let mut physical_load = graph_load.clone();
        let owns = |scope| match scope {
            WeightScope::Target => true,
            WeightScope::TargetSublayer(index)
            | WeightScope::TargetBranch {
                sublayer: index, ..
            } => range.blocks.contains(&(index.block as usize)),
            _ => false,
        };
        physical_load
            .target
            .retain(|weight| owns(weight.role.scope));
        physical_load
            .host_tables
            .retain(|table| owns(table.role.scope));
        physical_load
            .readout_candidates
            .retain(|weight| owns(weight.role.scope));
        physical_load
            .readout_choices
            .retain(|choice| owns(choice.scope));
        physical_load
            .unplaced_output
            .retain(|weight| owns(weight.role.scope));
        let weight_bytes = super::weight_bytes_by_component(&physical_load)?[0];
        // Qualification ends before import. Include the optional head's
        // source window, which is imported after initial state exists.
        let import_peak_bytes = super::source_import_peak_bytes(
            physical_load.weights().map(|weight| weight.source_bytes).max().unwrap_or(0),
        )?;
        Ok(DecoderRangeLoadPlan {
            range,
            decoder_blocks: self.ranges.last().unwrap().blocks.end,
            source_load: load.clone(),
            graph_load,
            physical_load,
            weight_bytes,
            import_peak_bytes,
        })
    }

    pub fn ranges(&self) -> &[DecoderRange] {
        &self.ranges
    }

    pub fn block_owner(&self, block: usize) -> Result<usize, String> {
        self.ranges
            .iter()
            .position(|range| range.blocks.contains(&block))
            .ok_or_else(|| format!("decoder block {block} is outside the placement"))
    }

    pub fn layer_owner(&self, layer: LayerRef) -> Result<usize, String> {
        match layer {
            LayerRef::Target(block) | LayerRef::Indexer(block) => self.block_owner(block as usize),
            LayerRef::Head(_) => Err("draft history needs its own device placement".into()),
        }
    }

    /// Split physical history domains before deriving slab geometry and graph
    /// contracts. Shared readers must execute on their source's owner: copying
    /// a residual alone cannot make remote history pointers valid.
    pub fn place_state(&self, layout: &ModelStateLayout) -> Result<PlacedStateLayout, String> {
        let mut placed = layout.clone();
        placed.target_history.clear();
        let mut history_owners = Vec::new();
        for domain in &layout.target_history {
            if let HistoryDomainLayout::Shared { source, layers } = domain {
                let owner = self.layer_owner(*source)?;
                for &reader in layers {
                    if self.layer_owner(reader)? != owner {
                        return Err("shared history crosses a decoder device boundary".into());
                    }
                }
                placed.target_history.push(domain.clone());
                continue;
            }
            let mut groups = vec![Vec::new(); self.ranges.len()];
            for component in domain.components() {
                groups[self.layer_owner(component.layer)?].push(component.clone());
            }
            for (owner, components) in groups.into_iter().enumerate() {
                if components.is_empty() {
                    continue;
                }
                let domain = match domain {
                    HistoryDomainLayout::Token { .. } => HistoryDomainLayout::Token { components },
                    HistoryDomainLayout::Window { rows, .. } => HistoryDomainLayout::Window {
                        rows: *rows,
                        components,
                    },
                    HistoryDomainLayout::Block { rate, .. } => HistoryDomainLayout::Block {
                        rate: *rate,
                        components,
                    },
                    HistoryDomainLayout::Shared { .. } => unreachable!(),
                };
                placed.target_history.push(domain);
                history_owners.push(owner);
            }
        }
        if layout.target_bank_layers.len() != layout.target_recurrent.len()
            || layout.target_banks.len() != layout.target_recurrent.len()
        {
            return Err("recurrent components differ from their semantic layer owners".into());
        }
        let recurrent_owners = layout
            .target_bank_layers
            .iter()
            .map(|&layer| self.layer_owner(layer))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PlacedStateLayout {
            layout: placed,
            history_owners,
            recurrent_owners,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::{BankComponent, CodecSpec, ComponentDescriptor};
    use seismic::DType;

    fn range(device: u8, blocks: Range<usize>) -> DecoderRange {
        DecoderRange {
            device: DeviceSelector::Cuda { uuid: [device; 16] },
            blocks,
        }
    }

    #[test]
    fn placement_rejects_gaps_overlaps_and_reused_devices() {
        for ranges in [
            vec![range(0, 0..2), range(1, 3..4)],
            vec![range(0, 0..3), range(1, 2..4)],
            vec![range(0, 0..2), range(0, 2..4)],
            vec![range(0, 0..2)],
        ] {
            assert!(DecoderPlacement::new(4, ranges).is_err());
        }
    }

    #[test]
    fn placement_preserves_global_history_and_recurrent_order() {
        let placement = DecoderPlacement::new(4, vec![range(0, 0..2), range(1, 2..4)]).unwrap();
        let components = [0, 2]
            .into_iter()
            .map(|block| {
                ComponentDescriptor::new(
                    LayerRef::Target(block),
                    CodecSpec::dense(DType::F32, 4, 4),
                    1,
                )
                .unwrap()
            })
            .collect();
        let bank = BankComponent::ConvWindow {
            width: 2,
            channels: 4,
            dtype: DType::F32,
        };
        let layout = ModelStateLayout {
            target_history: vec![HistoryDomainLayout::Token { components }],
            target_banks: vec![bank; 3],
            target_bank_layers: vec![
                LayerRef::Target(1),
                LayerRef::Target(3),
                LayerRef::Target(3),
            ],
            target_recurrent: vec![bank.spec(0).unwrap(); 3],
            head_history: vec![],
        };
        let placed = placement.place_state(&layout).unwrap();
        assert_eq!(placed.history_owners, [0, 1]);
        assert_eq!(placed.recurrent_owners, [0, 1, 1]);
        assert_eq!(placed.layout.target_recurrent, layout.target_recurrent);
        for (domain, block) in placed.layout.target_history.iter().zip([0, 2]) {
            assert_eq!(domain.components()[0].layer, LayerRef::Target(block));
        }
        let mut shared = layout;
        shared.target_history.push(HistoryDomainLayout::Shared {
            source: LayerRef::Target(0),
            layers: vec![LayerRef::Target(3)],
        });
        assert!(placement.place_state(&shared).is_err());
    }
}
