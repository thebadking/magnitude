//! Fixture-bound physical cache qualification; no cache latency or serving claim.
use super::*;
use magnitude_executor::{
    CpuExpertCache, CpuExpertLayerSpec, CpuExpertLease, platform::MemoryReserves,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

pub(super) fn fill(cpu: &Backend, expert: usize, outputs: &mut [Tensor; 3]) -> Result<(), String> {
    for kind in 0..3 {
        let source = cpu.raw[kind]
            .slice_leading(expert as u64, expert as u64 + 1)
            .map_err(why)?;
        cpu.imports[kind]
            .call_into(
                repack_weight::Args { source: &source },
                repack_weight::OutputArgs {
                    value: &mut outputs[kind],
                },
            )
            .map_err(why)?;
    }
    Ok(())
}

fn require_bound(cache: &CpuExpertCache) -> Result<Value, String> {
    let snapshot = cache.snapshot();
    require(
        snapshot
            .live_native_bytes
            .checked_add(snapshot.reserved_native_bytes)
            .is_some_and(|bytes| bytes <= snapshot.native_limit_bytes),
        "physical CPU cache exceeded its native limit",
    )?;
    serde_json::to_value(snapshot).map_err(why)
}

pub(super) fn contribution(
    cpu: &Backend,
    weights: &[Tensor; 3],
    expert: usize,
    input: usize,
    h: u64,
    f: u64,
) -> Result<Output, String> {
    contribution_from(cpu, weights, expert, &cpu.normalized[input], h, f)
}

pub(super) fn contribution_from(
    cpu: &Backend,
    weights: &[Tensor; 3],
    expert: usize,
    normalized: &Tensor,
    h: u64,
    f: u64,
) -> Result<Output, String> {
    let routes = Tensor::from_host(&cpu.device, Element::i32(), &[1, 1], &ints([expert as i32]))
        .map_err(why)?;
    let product = cpu
        .up
        .call(routed_gate_up::Args {
            normalized,
            routes: &routes,
            expert_gate: &weights[0],
            expert_up: &weights[1],
            activation: 0,
        })
        .map_err(why)?
        .value;
    let base = Tensor::zeros(&cpu.device, Element::f32(), &[1, h]).map_err(why)?;
    let coefficients =
        Tensor::from_host(&cpu.device, Element::f32(), &[1, 1], &1_f32.to_le_bytes())
            .map_err(why)?;
    let result = cpu
        .down
        .call(routed_down::Args {
            base: &base,
            product: &product.reshape(&[1, 1, f]).map_err(why)?,
            routes: &routes,
            weights: &coefficients,
            expert_down: &weights[2],
        })
        .map_err(why)?
        .value;
    Ok(Output { product, result })
}

fn packed_equal(cpu: &Backend, lease: &CpuExpertLease, expert: usize) -> Result<(), String> {
    for kind in 0..3 {
        let baseline = cpu.resident[kind]
            .slice_leading(expert as u64, expert as u64 + 1)
            .map_err(why)?
            .read_to_host()
            .map_err(why)?;
        require(
            lease.weights()[kind].read_to_host().map_err(why)? == baseline,
            "retained CPU cache bytes differ from native import",
        )?;
    }
    Ok(())
}

pub(super) fn qualify(
    cpu: &Backend,
    weights: &[Weight; 3],
    catalog: &DeviceCatalog,
) -> Result<Value, String> {
    let spec = CpuExpertLayerSpec {
        source_identity: serde_json::to_string(&weights.each_ref().map(|w| &w.identity))
            .map_err(why)?,
        experts: weights[0].shape[0] as usize,
        elements: weights.each_ref().map(|weight| weight.cpu),
        matrices: weights
            .each_ref()
            .map(|weight| [weight.shape[1], weight.shape[2]]),
    };
    let bytes = spec.record_bytes()?;
    use magnitude_kernels::routed_cpu_expert;
    let fused = prepare(
        routed_cpu_expert::native_entry_with(routed_cpu_expert::Elements {
            EGW: spec.elements[0],
            EUW: spec.elements[1],
            EDW: spec.elements[2],
        }),
        &cpu.device,
        &[("H", weights[0].shape[2]), ("F", weights[0].shape[1])],
    )?;
    let create = |records: u64| {
        CpuExpertCache::new(
            catalog.clone(),
            cpu.device.clone(),
            MemoryReserves::standard(),
            0,
            bytes
                .checked_mul(records)
                .ok_or("CPU cache qualification size overflow")?,
        )
    };
    let cache = create(2)?;
    let layer = cache.register_layer(spec.clone())?;
    require(
        cache.register_layer(spec.clone())? == layer,
        "shared source inventory did not deduplicate",
    )?;
    let mut rows = Vec::new();
    for expert in 0..spec.experts {
        let lease = cache
            .get_or_fill(layer, expert, |outputs| {
                let during = cache.snapshot();
                require(
                    during.pending_records == 1
                        && during.reserved_native_bytes == 0
                        && during.live_native_bytes >= bytes,
                    "backed CPU fill still reserves physical bytes twice",
                )?;
                require(
                    cache.get(layer, expert)?.is_none(),
                    "incomplete CPU triple became visible",
                )?;
                let invoked = AtomicUsize::new(0);
                let duplicate = cache.get_or_fill(layer, expert, |_| {
                    invoked.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })?;
                require(
                    duplicate.is_none() && invoked.load(Ordering::Relaxed) == 0,
                    "duplicate CPU fill was not coalesced",
                )?;
                fill(cpu, expert, outputs)
            })?
            .ok_or("physical CPU cache refused an admitted fixture record")?;
        packed_equal(cpu, &lease, expert)?;
        let input = expert % cpu.normalized.len();
        let expected = contribution(
            cpu,
            &cpu.resident,
            expert,
            input,
            weights[0].shape[2],
            weights[0].shape[1],
        )?;
        let observed = contribution(
            cpu,
            lease.weights(),
            0,
            input,
            weights[0].shape[2],
            weights[0].shape[1],
        )?;
        let product = difference(&expected.product, &observed.product)?;
        let result = difference(&expected.result, &observed.result)?;
        let held = lease.weights();
        let fused_result = fused
            .call(routed_cpu_expert::Args {
                normalized: &cpu.normalized[input],
                expert_gate: &held[0],
                expert_up: &held[1],
                expert_down: &held[2],
            })
            .map_err(why)?
            .value;
        require(
            fused_result.read_to_host().map_err(why)?
                == expected.result.read_to_host().map_err(why)?,
            "one-submission CPU expert differs from the original CPU operator control",
        )?;
        require(
            product["byte_equal"] == true && result["byte_equal"] == true,
            "CPU cache changed expert output",
        )?;
        rows.push(json!({"expert": expert, "product": product, "result": result, "cache": require_bound(&cache)?}));
    }
    let rotated = require_bound(&cache)?;
    require(
        rotated["retained_records"] == 2 && rotated["pending_records"] == 0,
        "two-record CPU cache did not enforce replacement",
    )?;

    // Two independent clients share the same capacity and read-lease rules.
    let (tx, rx) = mpsc::channel();
    let clients = std::thread::scope(|scope| -> Result<Vec<CpuExpertLease>, String> {
        for expert in [spec.experts - 2, spec.experts - 1] {
            let (cache, tx) = (cache.clone(), tx.clone());
            scope.spawn(move || {
                let _ = tx.send(cache.get(layer, expert));
            });
        }
        (0..2)
            .map(|_| {
                rx.recv_timeout(Duration::from_secs(2))
                    .map_err(why)??
                    .ok_or_else(|| "shared CPU cache client lost its retained record".into())
            })
            .collect()
    })?;
    let invoked = AtomicUsize::new(0);
    require(
        cache
            .get_or_fill(layer, 0, |_| {
                invoked.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })?
            .is_none()
            && invoked.load(Ordering::Relaxed) == 0,
        "leased CPU records were replaced or a refused fill ran",
    )?;
    let pinned = require_bound(&cache)?;
    drop(clients);

    let single = create(1)?;
    let single_layer = single.register_layer(spec.clone())?;
    require(
        single.get(layer, 0).is_err(),
        "CPU layer key crossed cache identities",
    )?;
    let lease = single
        .get_or_fill(single_layer, 0, |outputs| fill(cpu, 0, outputs))?
        .ok_or("single-record CPU cache fill refused")?;
    let alias = lease.weights()[0].clone();
    single.clear();
    let leased_after_clear = require_bound(&single)?;
    require(
        leased_after_clear["live_native_bytes"] == bytes
            && leased_after_clear["retired_records"] == 1,
        "clearing CPU cache released an in-flight read charge",
    )?;
    drop(lease);
    let alias_after_lease = require_bound(&single)?;
    let gate_bytes = weights[0]
        .cpu
        .canonical_byte_len(&[1, weights[0].shape[1], weights[0].shape[2]])
        .map_err(why)?;
    require(
        alias_after_lease["live_native_bytes"] == gate_bytes
            && alias_after_lease["retired_records"] == 1,
        "CPU tensor alias was no longer accounted after lease drop",
    )?;
    let invoked = AtomicUsize::new(0);
    require(
        single
            .get_or_fill(single_layer, 1, |_| {
                invoked.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })?
            .is_none()
            && invoked.load(Ordering::Relaxed) == 0,
        "retired CPU alias did not reserve its remaining bytes",
    )?;
    drop(alias);
    require(
        single.snapshot().live_native_bytes == 0,
        "CPU charge survived physical release",
    )?;

    let error = single.get_or_fill(single_layer, 2, |outputs| {
        let source = cpu.raw[0].slice_leading(2, 3).map_err(why)?;
        cpu.imports[0]
            .call_into(
                repack_weight::Args { source: &source },
                repack_weight::OutputArgs {
                    value: &mut outputs[0],
                },
            )
            .map_err(why)?;
        Err("cache qualification: interrupted triple".into())
    });
    require(
        error.is_err()
            && single.get(single_layer, 2)?.is_none()
            && single.snapshot().live_native_bytes == 0
            && single.snapshot().pending_records == 0,
        "failed CPU fill published a partial record or leaked its reservation",
    )?;
    let withdrawn = single.get_or_fill(single_layer, 3, |outputs| {
        fill(cpu, 3, outputs)?;
        single.clear();
        Ok(())
    })?;
    require(
        withdrawn.is_none()
            && single.get(single_layer, 3)?.is_none()
            && single.snapshot().live_native_bytes == 0
            && single.snapshot().pending_records == 0,
        "withdrawn CPU fill republished after cache clear",
    )?;
    let retry = single
        .get_or_fill(single_layer, 3, |outputs| fill(cpu, 3, outputs))?
        .ok_or("CPU cache did not recover after interrupted fills")?;
    packed_equal(cpu, &retry, 3)?;
    let recovered = require_bound(&single)?;

    let mut changed = spec.clone();
    changed.experts += 1;
    require(
        single.register_layer(changed).is_err(),
        "CPU source identity silently rebound its geometry",
    )?;
    let mut late = spec.clone();
    late.source_identity
        .push_str("/late-inventory-qualification");
    require(
        single.register_layer(late).is_err(),
        "CPU inventory grew after use",
    )?;

    let owner = create(1)?;
    let owner_layer = owner.register_layer(spec.clone())?;
    let owner_lease = owner
        .get_or_fill(owner_layer, 0, |outputs| fill(cpu, 0, outputs))?
        .ok_or("CPU owner lifetime fill refused")?;
    let weak_owner = Arc::downgrade(&owner);
    drop(owner);
    require(
        weak_owner.upgrade().is_some(),
        "CPU read lease lost its accounting owner",
    )?;
    drop(owner_lease);
    require(
        weak_owner.upgrade().is_none(),
        "CPU read lease introduced an ownership cycle",
    )?;

    Ok(
        json!({"status": "passed", "source_identity": spec.source_identity,
        "native_record_bytes": bytes, "qualified_experts": rows.len(), "rows": rows,
        "rotation": rotated, "shared_clients_pinned": pinned, "leased_after_clear": leased_after_clear,
        "alias_after_lease_drop": alias_after_lease, "recovered": recovered,
        "failed_fill_unpublished": true, "withdrawn_fill_unpublished": true,
        "builder_skipped_on_capacity_refusal": true, "pending_fill_coalesced": true,
        "source_inventory_sealed": true, "cross_cache_key_refused": true,
        "lease_retains_cache_owner": true,"single_submission_cpu_exact":true,"cpu_expert_kernel":kernel_identity(&fused),
        "scope": "Physical cache/storage and CPU operator qualification on one real weight triple and fixture readout vectors; no serving dispatch, latency or model-quality claim"}),
    )
}
