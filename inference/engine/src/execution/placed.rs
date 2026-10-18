//! Paired startup: each owner admits and imports its local physical inventory.
use super::*;
use magnitude_executor::{AllocatedResources, DecoderPlacement, ResidentTarget};

struct Owner {
    execution: Rc<magnitude_executor::ExecutionPlan>,
    device: Rc<seismic::Device>,
    programs: Rc<AttestedPrograms>,
    resources: AllocatedResources,
    resident: ResidentTarget,
    startup: StartupClaims,
}

pub(super) fn build_placed(
    catalog: DeviceCatalog,
    manifest: &ExecutionManifest,
    package: Arc<Package>,
    unprepared: UnpreparedUnits,
    progress: Rc<dyn Fn(LoadProgress)>,
) -> Result<LoadAttempt<NativeDomain>, LoadError> {
    let placement = DecoderPlacement::new(
        manifest.definition.decoder.blocks.len(),
        manifest.decoder_ranges.clone(),
    )
    .map_err(internal)?;
    if placement.ranges().len() != 2
        || manifest.model.method != ResolvedMethod::Plain
        || manifest.definition.vision.is_some()
    {
        return Err(internal(
            "two-device startup requires two ranges and plain text execution",
        ));
    }
    let catalog = Rc::new(catalog);
    let mut prepared = Vec::new();
    for range in placement.ranges() {
        let mut local = manifest.clone();
        local.device = DeviceRequest::Selector(range.device);
        match prepare(catalog.clone(), &local, &package, unprepared, progress.clone())? {
            LoadAttempt::Loaded(owner) => prepared.push(owner),
            LoadAttempt::Unprepared => return Ok(LoadAttempt::Unprepared),
        }
    }
    let limits = prepared[0].draft.policy().limits();
    if prepared
        .iter()
        .any(|owner| owner.draft.policy().limits() != limits)
    {
        return Err(internal("decoder owners disagree on launch limits"));
    }
    let source = prepared[0]
        .range
        .as_ref()
        .ok_or_else(|| internal("first owner has no range inventory"))?
        .source_load();
    let capacities = prepared
        .iter()
        .map(|owner| owner.capacity)
        .collect::<Vec<_>>();
    let state = ResourcePlanner::state_plan_placed(
        &manifest.definition,
        source,
        manifest.model.kv_codec,
        limits,
        &placement,
        &capacities,
    )
    .map_err(internal)?;
    let protected_host_bytes = source.host_table_bytes().map_err(internal)?;
    let mut targets = Vec::new();
    let mut readouts = Vec::new();
    for owner in &mut prepared {
        let range = owner
            .range
            .as_ref()
            .ok_or_else(|| internal("owner has no range inventory"))?;
        targets.push(
            owner
                .programs
                .prepare_target_graphs_range(
                    owner.opened.device(),
                    owner.draft.load(),
                    &manifest.definition.decoder,
                    &state,
                    owner.draft.programs().target(),
                    limits,
                    &owner.workers,
                    range.range().blocks.clone(),
                )
                .map_err(internal)?,
        );
        readouts.push(
            owner
                .programs
                .prepare_target_readout_graphs(
                    owner.opened.device(),
                    owner.draft.load(),
                    &manifest.definition.decoder,
                    limits,
                )
                .map_err(internal)?,
        );
        owner
            .programs
            .prepare_auxiliary_graphs(
                owner.opened.device(),
                owner.draft.load(),
                &manifest.definition,
                owner.draft.programs(),
                &state,
                limits,
                owner.draft.policy().method(),
            )
            .map_err(internal)?;
    }
    if manifest.model.lookahead && !targets[0].supports_selected_return() {
        return Err(internal("placed lookahead requires admitted selected-token return storage"));
    }
    let mut plans = Vec::new();
    for (index, owner) in prepared.iter().enumerate() {
        let plan = ResourcePlanner::plan_range_with_state(
            owner.backend,
            platform::shared_role(owner.opened.device().info())
                .map_err(|error| internal(error.to_string()))?,
            &state,
            owner.range.as_ref().unwrap(),
            &targets[index],
            &readouts[index],
            owner
                .programs
                .state_graphs()
                .ok_or_else(|| internal("owner has no state graphs"))?,
        )
        .map_err(internal)?;
        plans.push(if index == 0 {
            plan
        } else {
            plan.with_boundary_from(&targets[0], &targets[1])
                .map_err(internal)?
        });
    }
    let host_experts = if prepared[0].backend == BackendName::Cuda {
        magnitude_executor::HostExpertCache::new(catalog.as_ref().clone(), manifest.reserves,
            protected_host_bytes).map_err(internal)?
    } else { None };
    let mut owners = Vec::new();
    for (((owner, target), readout), plan) in
        prepared.into_iter().zip(targets).zip(readouts).zip(plans)
    {
        let PreparedPrograms {
            mut startup,
            draft,
            opened,
            mut programs,
            backend,
            workers,
            range,
            ..
        } = owner;
        let range = range.unwrap();
        programs.install_target_graphs(target);
        programs.install_target_readout_graphs(readout);
        let execution = Rc::new(
            draft
                .admit_range(&range, plan.clone())
                .map_err(|error| LoadError::from(classify_plan(error, backend)))?,
        );
        opened.device().staff(None);
        drop(workers);
        let selector = opened.selector();
        let device = Rc::new(opened.into_device());
        let programs = Rc::new(programs);
        let identity = ResourceDomainId::new(format!("{}:{selector}", manifest.package.identity))
            .map_err(internal)?;
        startup.claim_graphs("range graph and transfer buffers", &plan)?;
        let resources = ResourceAllocator::allocate_range(
            &plan,
            &device,
            identity.clone(),
            programs.target_graphs().unwrap(),
            programs.target_readout_graphs().unwrap(),
            programs.state_graphs().unwrap(),
        )
        .map_err(|error| internal(error.to_string()))?;
        startup.hold_host_tables(range.physical_load().host_table_bytes().map_err(internal)?)?;
        let shared = if platform::shared_role(device.info())
            .map_err(|error| internal(error.to_string()))?
            == DomainRole::Staging
        {
            range
                .physical_load()
                .shared_weight_bytes()
                .map_err(internal)?
        } else {
            0
        };
        let peak = range
            .weight_bytes()
            .checked_add(range.import_peak_bytes())
            .and_then(|bytes| bytes.checked_sub(shared))
            .ok_or_else(|| internal("range import charge overflow"))?;
        let staged = range
            .import_peak_bytes()
            .checked_add(shared)
            .ok_or_else(|| internal("range staging overflow"))?;
        startup.claim("range target import", peak, staged)?;
        let mut residency =
            ResidencyStore::new_range(device.clone(), programs.clone(), &range, identity)
                .map_err(|error| internal(error.to_string()))?;
        if let Some(cache) = &host_experts { residency.keep_host_experts(cache.clone()); }
        if let Some(cache) = &manifest.kernel_cache {
            residency.keep_table_rows(cache.join("table-rows"));
        }
        let observer_catalog = catalog.clone();
        let observer_device = device.clone();
        let observer_progress = progress.clone();
        let reserves = manifest.reserves;
        let mut observed = Instant::now();
        let resident = residency
            .load_target(
                &manifest.definition,
                &package,
                Box::new(move |completed_bytes, total_bytes, checkpoint| {
                    observer_progress(LoadProgress::ImportingWeights {
                        completed_bytes,
                        total_bytes,
                    });
                    if checkpoint || observed.elapsed() >= IMPORT_OBSERVATION_INTERVAL {
                        observed = Instant::now();
                        require_normal(&observer_catalog, &observer_device, &reserves)?;
                    }
                    Ok(())
                }),
            )
            .map_err(|error| internal(error.to_string()))?;
        owners.push(Owner {
            execution,
            device,
            programs,
            resources,
            resident,
            startup,
        });
    }
    let initial = state.startup_state_bytes_by_owner().map_err(internal)?;
    for (owner, bytes) in owners.iter_mut().zip(initial) {
        owner.startup.claim("placed initial state", bytes, 0)?;
    }
    let devices = owners
        .iter()
        .map(|owner| owner.device.clone())
        .collect::<Vec<_>>();
    let target_state = state
        .allocate_target_state_placed(&devices)
        .map_err(internal)?;
    for owner in &mut owners {
        let constants = owner
            .programs
            .target_graphs()
            .unwrap()
            .binding_constant_bytes()
            .map_err(internal)?;
        owner
            .startup
            .claim("range binding constants", constants, 0)?;
    }
    let second = owners.pop().unwrap();
    let first = owners.pop().unwrap();
    let owner_plans = vec![(first.device.info().selector, first.execution.resources().clone()),
        (second.device.info().selector, second.execution.resources().clone())];
    let plan = first.execution.resources().clone();
    let device = first.device.info().selector;
    let (mut domain, bindings) = ExecutorDomain::new_placed(
        first.execution,
        Rc::new(manifest.definition.clone()),
        first.startup.into_heap(),
        first.programs,
        first.resources,
        first.resident,
        target_state,
        second.execution,
        second.startup.into_heap(),
        second.programs,
        second.resources,
        second.resident,
    )
    .map_err(internal)?;
    domain.register_allocated_holdings().map_err(internal)?;
    let bindings = warm_up(&mut domain, bindings, false)?;
    Ok(LoadAttempt::Loaded(NativeDomain {
        domain,
        bindings,
        plan,
        owner_plans,
        device,
    }))
}
