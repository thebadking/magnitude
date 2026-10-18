use super::*;
use crate::programs::native_target_graph::BankPart;
use crate::streaming::{
    layout::{bank_port, BankTensors},
    shared_bank::SharedBankStore,
};
use magnitude_family_contracts::SublayerIndex;
use seismic::{DeviceCatalog, Tensor};

fn resident_plan() -> ModelLoadPlan {
    resident_plan_of([4, 4, 8])
}

/// Three routed layers whose expert matrices are `[2, columns[block]]`.
fn resident_plan_of(columns: [u64; 3]) -> ModelLoadPlan {
    let target = (0..3)
        .flat_map(|block| {
            BANK_KINDS.into_iter().enumerate().map(move |(kind, name)| {
                let scope = WeightScope::TargetSublayer(SublayerIndex { block, sublayer: 1 });
                let shape = vec![256, 2, columns[block as usize]];
                let element = Element::f32();
                let bytes = Tensor::canonical_layout(element, &shape).unwrap().0;
                WeightPlan {
                    role: WeightRole { scope, kind: name },
                    component: ArtifactComponent {
                        kind: ArtifactComponentKind::Target,
                        identity: ArtifactIdentity([0; 32]),
                    },
                    source: element,
                    upload: element,
                    resident: element,
                    descriptor: WeightDescriptor::stored(
                        format!("blk.{block}.kind.{kind}"),
                        shape.clone(),
                    ),
                    shape,
                    source_bytes: bytes,
                    resident_bytes: bytes,
                    scale: None,
                    bound: 0,
                    streamed: None,
                }
            })
        })
        .collect();
    ModelLoadPlan {
        target,
        head: None,
        vision: None,
        host_tables: vec![],
        readout_candidates: vec![],
        readout_choices: vec![],
        unplaced_output: vec![],
        residual_conditioning: None,
        bank: None,
    }
}

#[test]
fn shared_storage_shapes_ports_and_physical_charge_agree() {
    let resident = resident_plan();
    let comparison = resident.clone().with_expert_bank(9, 1).unwrap();
    let shared = resident.with_shared_expert_bank(9, 1).unwrap();
    let bank = shared.expert_bank().unwrap();
    let groups = bank.shared.as_ref().unwrap();
    assert_eq!(groups.groups.len(), 2);
    assert_eq!(groups.groups[0].records, 18);
    assert_eq!(groups.groups[1].records, 9);
    for weight in shared.target() {
        let records = bank.records(weight.role.scope).unwrap();
        assert_eq!(weight.shape[0], records);
        assert_eq!(bank.routed(weight.role.scope).unwrap().slots, records);
        for port in [BankPart::Stamps, BankPart::Commits] {
            assert_eq!(
                bank_port(&shared, weight.role.scope, port).unwrap().1,
                vec![records]
            );
        }
        assert_eq!(
            bank_port(&shared, weight.role.scope, BankPart::CopyList { kind: 0 })
                .unwrap()
                .1,
            vec![2 + 2 * bank.experts]
        );
        assert_eq!(bank.resident_plan(weight.role).unwrap().shape[0], 256);
        assert_eq!(weight.source_bytes, 0);
    }
    assert!(comparison.expert_bank().unwrap().shared.is_none());
    let scope = bank.layers[0].scope;
    assert!(bank_port(&comparison, scope, BankPart::Commits).is_err());
    assert_eq!(
        weight_bytes_by_component(&shared).unwrap(),
        [groups.allocated_bytes + bank.charged_bytes().unwrap(), 0, 0]
    );
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let baseline = device.memory_usage().charged;
    let mut store = SharedBankStore::new(bank).unwrap();
    let tensors = shared
        .target()
        .iter()
        .map(|weight| store.bind(&device, weight.role).unwrap())
        .collect::<Vec<_>>();
    assert!(tensors[0].shares_allocation(&tensors[3]));
    assert!(!tensors[0].shares_allocation(&tensors[6]));
    let lender = bank
        .conveyor
        .lender
        .map(|group| store.arenas(&device, group).unwrap());
    let controls = BankTensors::allocate(&device, bank, groups, lender, &mut || Ok(())).unwrap();
    assert_eq!(
        device.memory_usage().charged - baseline,
        weight_bytes_by_component(&shared).unwrap()[0]
    );
    drop(controls);
    drop(tensors);
    drop(store);
    assert_eq!(device.memory_usage().charged, baseline);
}

#[test]
fn shared_formation_preserves_source_geometry_and_rejects_double_banking() {
    let original = resident_plan();
    let shared = original.clone().with_shared_expert_bank(9, 1).unwrap();
    let bank = shared.expert_bank().unwrap();
    for weight in original.target() {
        assert_eq!(bank.resident_plan(weight.role).unwrap(), weight);
    }
    assert!(shared.with_shared_expert_bank(9, 1).is_err());
    assert!(original.clone().with_shared_expert_bank(0, 1).is_err());
    assert!(original.with_shared_expert_bank(256, 1).is_err());
}

#[test]
fn group_minimum_is_one_launch_union_and_exact_charge_includes_padding() {
    let load = resident_plan()
        .with_shared_expert_bank_capacity(&[8, 17], 1)
        .unwrap();
    let bank = load.expert_bank().unwrap();
    let shared = bank.shared.as_ref().unwrap();
    assert_eq!(shared.groups[0].layers.len(), 2);
    assert_eq!(shared.groups[0].records, 8); // two layers share ONE minimum
    assert_eq!(shared.groups[1].records, 17);
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let baseline = device.memory_usage().charged;
    let charge = weight_bytes_by_component(&load).unwrap()[0];
    device.set_memory_limit(Some(baseline + charge));
    let mut store = SharedBankStore::new(bank).unwrap();
    let weights = load
        .target()
        .iter()
        .map(|w| store.bind(&device, w.role).unwrap())
        .collect::<Vec<_>>();
    let lender = bank
        .conveyor
        .lender
        .map(|group| store.arenas(&device, group).unwrap());
    let controls = BankTensors::allocate(&device, bank, shared, lender, &mut || Ok(())).unwrap();
    assert_eq!(weights[0].extents(), &[8, 2, 4]);
    assert_eq!(weights[6].extents(), &[17, 2, 8]);
    assert_eq!(device.memory_usage().charged, baseline + charge);
    drop(controls);
    drop(weights);
    drop(store);
    assert_eq!(device.memory_usage().charged, baseline);
}

#[test]
fn group_capacity_rejects_unions_it_cannot_hold_and_foreign_metadata() {
    for records in [&[7, 8][..], &[8, 7], &[513, 8], &[8, 257], &[8], &[8, 8, 8]] {
        assert!(resident_plan()
            .with_shared_expert_bank_capacity(records, 1)
            .is_err());
    }
    assert!(resident_plan()
        .with_shared_expert_bank_capacity(&[8, 8], 0)
        .is_err());
    assert!(resident_plan()
        .with_shared_expert_bank_capacity(&[8, 8], 257)
        .is_err());
    let load = resident_plan()
        .with_shared_expert_bank_capacity(&[64, 64], 8)
        .unwrap();
    let mut bank = load.expert_bank().unwrap().clone();
    bank.shared.as_mut().unwrap().groups[0].allocated_bytes += 1;
    assert!(SharedBankStore::new(&bank).is_err());
    bank = load.expert_bank().unwrap().clone();
    bank.shared.as_mut().unwrap().group_of_layer.swap(0, 2);
    assert!(SharedBankStore::new(&bank).is_err());
}

/// A bank group holding the conveyor beside a launch's union lends it its
/// last records (spec 4.9): the load charges no conveyor bytes, every
/// conveyor view lies on those records, and the device holds exactly the
/// plan's charge. A group that cannot lends nothing and the load holds the
/// conveyor's arenas.
#[test]
fn a_bank_group_with_records_to_spare_lends_the_conveyor_its_bytes() {
    // One encoding, one pool per routed tensor: its conveyor is a whole
    // layer of 32-byte records, 256 records of the group.
    let held = resident_plan_of([4, 4, 4])
        .with_shared_expert_bank_capacity(&[256], 1)
        .unwrap();
    let held = held.expert_bank().unwrap();
    assert_eq!(held.conveyor.lender, None);
    let conveyor = (0..3)
        .map(|kind| held.conveyor_arena_bytes(kind).unwrap())
        .sum::<u64>();
    assert_eq!(conveyor, 3 * 256 * 32);
    let load = resident_plan_of([4, 4, 4])
        .with_shared_expert_bank_capacity(&[300], 1)
        .unwrap();
    let bank = load.expert_bank().unwrap();
    assert_eq!(bank.conveyor.lender, Some(0));
    assert_eq!(bank.conveyor.lent, 256);
    assert_eq!(bank.conveyor.offsets, vec![44 * 32; 3]);
    assert_eq!(
        bank.staging_bytes().unwrap() + conveyor,
        held.staging_bytes().unwrap()
    );
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Cpu)
        .unwrap();
    let baseline = device.memory_usage().charged;
    let mut store = SharedBankStore::new(bank).unwrap();
    let weights = load
        .target()
        .iter()
        .map(|w| store.bind(&device, w.role).unwrap())
        .collect::<Vec<_>>();
    let lender = store.arenas(&device, 0).unwrap();
    let shared = bank.shared.as_ref().unwrap();
    let controls =
        BankTensors::allocate(&device, bank, shared, Some(lender.clone()), &mut || Ok(())).unwrap();
    for (pool, view) in controls.conveyors.iter().enumerate() {
        assert!(view.shares_allocation(&lender[bank.pools[pool].kind]));
    }
    assert_eq!(
        device.memory_usage().charged - baseline,
        weight_bytes_by_component(&load).unwrap()[0]
    );
    drop((controls, weights, lender, store));
    assert_eq!(device.memory_usage().charged, baseline);
}

#[test]
fn cuda_streaming_staging_is_host_visible_and_charged_separately() {
    let catalog = DeviceCatalog::discover().unwrap();
    let Ok(device) = catalog.open_backend(seismic::BackendName::Cuda) else {
        return;
    };
    let lent = resident_plan_of([4, 4, 4])
        .with_shared_expert_bank_capacity(&[320], 1)
        .unwrap();
    assert!(lent.expert_bank().unwrap().conveyor.lender.is_some());
    let load = lent.with_host_expert_staging(false).unwrap();
    let bank = load.expert_bank().unwrap();
    assert!(bank.conveyor.lender.is_none());
    assert_eq!(
        load.shared_weight_bytes().unwrap(),
        bank.shared_bytes().unwrap()
    );
    assert_eq!(
        bank.charged_bytes().unwrap(),
        bank.shared_bytes().unwrap() + bank.scratch_bytes().unwrap()
    );
    let baseline = device.memory_usage().charged;
    let baseline_held = device.memory_usage().held;
    // The initial grant covers the host phase only. Scratch must wait for
    // the owner to refresh the grant after all shared storage exists.
    device.set_memory_limit(Some(baseline + bank.shared_bytes().unwrap()));
    let mut checkpoints = 0;
    let tensors = BankTensors::allocate(
        &device,
        bank,
        bank.shared.as_ref().unwrap(),
        None,
        &mut || {
            checkpoints += 1;
            assert_eq!(device.memory_usage().held, baseline_held);
            assert_eq!(
                device.memory_usage().charged,
                baseline + bank.shared_bytes().unwrap()
            );
            // This synthetic owner grants the remaining physical headroom;
            // CUDA may reserve an allocator block larger than these tensors.
            device.set_memory_limit(None);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(checkpoints, 1);
    device.set_memory_limit(None);
    assert_eq!(
        device.memory_usage().charged - baseline,
        bank.charged_bytes().unwrap()
    );
    for tensor in [&tensors.step, &tensors.heartbeat]
        .into_iter()
        .chain(tensors.pools.iter())
        .chain(tensors.conveyors.iter())
    {
        assert!(tensor.host_window().is_ok());
    }
    for layer in &tensors.layers {
        for tensor in [
            &layer.slot_map,
            &layer.stamps,
            &layer.commits,
            &layer.control,
            &layer.placement,
            &layer.scratch_control,
            &layer.rings[0],
            &layer.rings[1],
        ] {
            assert!(tensor.host_window().is_ok());
        }
    }
    assert!(tensors
        .scratch
        .iter()
        .all(|tensor| tensor.host_window().is_err()));
}

#[test]
fn decoder_ranges_have_independent_cache_groups_and_compact_pools() {
    let load = resident_plan().with_shared_expert_bank(9, 1).unwrap();
    let bank = load.expert_bank().unwrap();
    let left = bank.for_decoder_range(0..2, Some(&[32])).unwrap().unwrap();
    let right = bank.for_decoder_range(2..3, Some(&[16])).unwrap().unwrap();
    assert_eq!(left.layers.len(), 2);
    assert_eq!(right.layers.len(), 1);
    for (local, records) in [(&left, 32), (&right, 16)] {
        let shared = crate::streaming::shared_bank::SharedBankPlan::admitted(local).unwrap();
        assert_eq!(shared.groups.len(), 1);
        assert_eq!(shared.groups[0].records, records);
        assert_eq!(local.pools.len(), 3);
        assert_eq!(local.scratch.len(), 3);
        for layer in &local.layers {
            for kind in 0..3 {
                let original = bank.layers.iter().find(|old| old.scope == layer.scope).unwrap();
                assert_eq!(local.pools[layer.tensors[kind].pool], bank.pools[original.tensors[kind].pool]);
                assert_eq!(local.scratch[layer.tensors[kind].scratch], bank.scratch[original.tensors[kind].scratch]);
            }
        }
    }
    assert_eq!(right.layers[0].scope, bank.layers[2].scope);
    assert!(bank.for_decoder_range(3..4, None).unwrap().is_none());
    assert!(bank.for_decoder_range(0..2, Some(&[1])).is_err());
    assert_eq!(bank.shared.as_ref().unwrap().groups[0].records, 18);
}
