//! A real CUDA/CPU handshake over fixture-bound expert work, not serving dispatch.
use super::*;
use magnitude_executor::{
    CpuExpertCache, CpuExpertExchange, CpuExpertLayerSpec, platform::MemoryReserves,
};
use magnitude_kernels::routed_cpu_exchange;
use std::time::Duration;

fn launch(
    kernel: &NativeKernel<routed_cpu_exchange::Entry>,
    exchange: &CpuExpertExchange,
    input: &Tensor,
    sequence: i32,
) -> Result<Tensor, String> {
    let step =
        Tensor::from_host(&input.device(), Element::i32(), &[1], &ints([sequence])).map_err(why)?;
    let mut activation = exchange.activation().clone();
    let mut control = exchange.control().clone();
    kernel
        .call(routed_cpu_exchange::Args {
            normalized: input,
            activation: &mut activation,
            contribution: exchange.contribution(),
            step: &step,
            control: &mut control,
            heartbeat: exchange.heartbeat(),
        })
        .map(|output| output.value)
        .map_err(why)
}

fn request(
    exchange: &Arc<CpuExpertExchange>,
) -> Result<magnitude_executor::CpuExpertRequest, String> {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        exchange.pulse();
        if let Some(request) = exchange.try_request()? {
            return Ok(request);
        }
        if Instant::now() >= deadline {
            exchange.abort();
            return Err("GPU did not publish the CPU activation within the bound".into());
        }
        std::thread::sleep(Duration::from_micros(25));
    }
}

fn zero(tensor: &Tensor) -> Result<bool, String> {
    Ok(tensor
        .read_to_host()
        .map_err(why)?
        .iter()
        .all(|&byte| byte == 0))
}

pub(super) fn qualify(
    cpu: &Backend,
    gpu: &Backend,
    weights: &[Weight; 3],
    catalog: &DeviceCatalog,
    peer_selector: &str,
    device_options: DeviceOptions,
) -> Result<Value, String> {
    let selector: DeviceSelector = serde_json::from_str(peer_selector).map_err(why)?;
    let peer = catalog
        .open_with(catalog.resolve(selector).map_err(why)?, device_options)
        .map_err(why)?;
    require(
        peer.backend() == BackendName::Cuda,
        "exchange peer must be CUDA",
    )?;
    require(
        serde_json::to_value(peer.info().selector).map_err(why)?
            != serde_json::to_value(gpu.device.info().selector).map_err(why)?,
        "exchange qualification requires two distinct physical GPUs",
    )?;
    peer.set_memory_limit(Some(64 << 20));
    let h = weights[0].shape[2];
    let f = weights[0].shape[1];
    let spec = CpuExpertLayerSpec {
        source_identity: serde_json::to_string(&weights.each_ref().map(|w| &w.identity))
            .map_err(why)?,
        experts: weights[0].shape[0] as usize,
        elements: weights.each_ref().map(|w| w.cpu),
        matrices: weights.each_ref().map(|w| [w.shape[1], w.shape[2]]),
    };
    let cache = CpuExpertCache::new(
        catalog.clone(),
        cpu.device.clone(),
        MemoryReserves::standard(),
        0,
        spec.record_bytes()?
            .checked_mul(2)
            .ok_or("exchange cache capacity overflow")?,
    )?;
    let layer = cache.register_layer(spec.clone())?;
    let devices = [&gpu.device, &peer];
    let kernels = devices
        .iter()
        .map(|device| prepare(routed_cpu_exchange::native_entry(), device, &[("H", h)]))
        .collect::<Result<Vec<_>, _>>()?;
    let channels = devices
        .iter()
        .map(|device| CpuExpertExchange::new(device, h))
        .collect::<Result<Vec<_>, _>>()?;
    let feature_bytes = cpu
        .normalized
        .iter()
        .map(|input| input.read_to_host().map_err(why))
        .collect::<Result<Vec<_>, _>>()?;
    require(
        feature_bytes.len() >= 2 && feature_bytes[0] != feature_bytes[1],
        "exchange lifecycle requires two distinct retained fixture rows",
    )?;
    let peer_inputs = feature_bytes
        .iter()
        .map(|bytes| Tensor::from_host(&peer, Element::bf16(), &[1, h], bytes).map_err(why))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = [&gpu.normalized, &peer_inputs];
    let mut rows = Vec::new();
    for expert in 0..spec.experts {
        let owner = expert % 2;
        let sequence = (expert / 2 + 1) as i32;
        let input = expert % feature_bytes.len();
        let expected = cache::contribution(cpu, &cpu.resident, expert, input, h, f)?
            .result
            .read_to_host()
            .map_err(why)?;
        let lease = cache
            .get_or_fill(layer, expert, |outputs| cache::fill(cpu, expert, outputs))?
            .ok_or("exchange cache refused its next real expert")?;
        let channel = channels[owner].clone();
        let (observed, publication) = std::thread::scope(|scope| -> Result<_, String> {
            let channel = channel.clone();
            let expected_activation = &feature_bytes[input];
            let worker = scope.spawn(move || -> Result<Value, String> {
                let request = request(&channel)?;
                require(request.sequence() == sequence, "GPU published an unexpected sequence")?;
                require(request.activation() == expected_activation, "GPU changed the BF16 activation bytes")?;
                require(channel.try_request()?.is_none(), "a second host consumer claimed the same request")?;
                let normalized = Tensor::from_host(&cpu.device, Element::bf16(), &[1, h], request.activation()).map_err(why)?;
                let result = cache::contribution_from(cpu, lease.weights(), 0, &normalized, h, f)?.result;
                let contribution = result.read_to_host().map_err(why)?;
                // Native work has physically completed before publication. The
                // retained expert lease remains live throughout that work.
                request.complete(&contribution)?;
                drop(lease);
                Ok(json!({"activation_byte_equal": true, "single_consumer": true, "sequence": sequence}))
            });
            let output = launch(
                &kernels[owner],
                &channels[owner],
                &inputs[owner][input],
                sequence,
            );
            let publication = worker
                .join()
                .map_err(|_| "CPU exchange worker panicked")??;
            Ok((output?.read_to_host().map_err(why)?, publication))
        })?;
        require(
            observed == expected,
            "GPU received a different CPU expert contribution",
        )?;
        require(
            channels[owner].failure().is_none(),
            "successful exchange left a failure",
        )?;
        rows.push(json!({"expert": expert, "owner": owner, "contribution_byte_equal": true, "publication": publication}));
    }

    // Two active owners consume the shared cache's last two retained experts.
    // Their physical CPU pool serializes its own submissions; this is ownership
    // and publication coverage, not evidence of parallel CPU speed.
    let concurrent_ready = std::sync::atomic::AtomicUsize::new(0);
    let concurrent = std::thread::scope(|scope| -> Result<Value, String> {
        let mut workers = Vec::new();
        let ready = &concurrent_ready;
        for owner in 0..2 {
            let expert = spec.experts - 2 + owner;
            let lease = cache
                .get(layer, expert)?
                .ok_or("shared exchange expert was not retained")?;
            let channel = channels[owner].clone();
            let expected = &feature_bytes[owner];
            workers.push(scope.spawn(move || -> Result<(), String> {
                let request = request(&channel)?;
                require(
                    request.activation() == expected,
                    "concurrent activation mismatch",
                )?;
                ready.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                let deadline = Instant::now() + Duration::from_secs(3);
                while ready.load(std::sync::atomic::Ordering::Acquire) != 2 {
                    channel.pulse();
                    if Instant::now() >= deadline {
                        return Err("second GPU did not publish concurrently".into());
                    }
                    std::thread::sleep(Duration::from_micros(25));
                }
                let normalized =
                    Tensor::from_host(&cpu.device, Element::bf16(), &[1, h], request.activation())
                        .map_err(why)?;
                let result =
                    cache::contribution_from(cpu, lease.weights(), 0, &normalized, h, f)?.result;
                request.complete(&result.read_to_host().map_err(why)?)
            }));
        }
        let mut launched = Vec::new();
        for owner in 0..2 {
            let kernel = &kernels[owner];
            let channel = &channels[owner];
            let input = &inputs[owner][owner];
            launched.push(scope.spawn(move || launch(kernel, channel, input, 100)));
        }
        let mut returned = Vec::new();
        for launch in launched {
            returned.push(
                launch
                    .join()
                    .map_err(|_| "GPU exchange launcher panicked")??,
            );
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| "concurrent CPU worker panicked")??;
        }
        require(
            ready.load(std::sync::atomic::Ordering::Acquire) == 2,
            "both GPU requests were not observed",
        )?;
        for owner in 0..2 {
            let expected =
                cache::contribution(cpu, &cpu.resident, spec.experts - 2 + owner, owner, h, f)?
                    .result;
            require(
                returned[owner].read_to_host().map_err(why)?
                    == expected.read_to_host().map_err(why)?,
                "concurrent contribution mismatch",
            )?;
            require(
                channels[owner].failure().is_none(),
                "concurrent channel failed",
            )?;
        }
        Ok(
            json!({"active_gpu_owners": 2, "shared_cpu_cache": true, "contributions_byte_equal": true}),
        )
    })?;

    let mut lifecycle = Vec::new();
    for owner in 0..2 {
        let channel = CpuExpertExchange::new(devices[owner], h)?;
        let returned = std::thread::scope(|scope| -> Result<Tensor, String> {
            let worker_channel = channel.clone();
            let worker = scope.spawn(move || -> Result<(), String> {
                let request = request(&worker_channel)?;
                require(
                    request.complete(&[0]).is_err(),
                    "invalid contribution length was accepted",
                )
            });
            let returned = launch(&kernels[owner], &channel, &inputs[owner][0], 1);
            worker.join().map_err(|_| "failure CPU worker panicked")??;
            returned
        })?;
        require(
            zero(&returned)? && channel.failure().is_some(),
            "failed exchange did not return defined zero storage",
        )?;
        require(
            zero(&launch(&kernels[owner], &channel, &inputs[owner][1], 2)?)?,
            "poisoned successor read a stale result",
        )?;
        require(
            channel.try_request().is_err(),
            "poisoned channel allowed another host request",
        )?;
        require(
            channel.activation().read_to_host().map_err(why)? == feature_bytes[0],
            "poisoned successor overwrote the earlier activation payload",
        )?;
        let withdrawn = CpuExpertExchange::new(devices[owner], h)?;
        let returned = std::thread::scope(|scope| -> Result<Tensor, String> {
            let worker_channel = withdrawn.clone();
            let worker = scope.spawn(move || -> Result<(), String> {
                let request = request(&worker_channel)?;
                worker_channel.abort();
                require(
                    request.complete(&vec![0; h as usize * 4]).is_err(),
                    "withdrawn request published a response",
                )
            });
            let returned = launch(&kernels[owner], &withdrawn, &inputs[owner][0], 1);
            worker
                .join()
                .map_err(|_| "withdrawal CPU worker panicked")??;
            returned
        })?;
        require(
            zero(&returned)? && withdrawn.failure().is_some(),
            "withdrawn exchange did not fail with defined output storage",
        )?;
        let cancelled = CpuExpertExchange::new(devices[owner], h)?;
        cancelled.abort();
        require(
            zero(&launch(&kernels[owner], &cancelled, &inputs[owner][0], 1)?)?
                && cancelled.failure() == Some(-1),
            "prepublication withdrawal was lost",
        )?;
        let stale = launch(&kernels[owner], &channels[owner], &inputs[owner][0], 100)?;
        require(
            zero(&stale)? && channels[owner].failure().is_some(),
            "repeated sequence reused an earlier acknowledgement",
        )?;
        let abandoned = CpuExpertExchange::new(devices[owner], h)?;
        require(
            zero(&launch(&kernels[owner], &abandoned, &inputs[owner][0], 1)?)?,
            "abandoned exchange did not terminate with zero storage",
        )?;
        require(
            abandoned.failure().is_some(),
            "dead-host watchdog did not poison the channel",
        )?;
        lifecycle.push(json!({"owner": owner, "invalid_response_poisoned": true, "poisoned_successor_refused": true,
            "poisoned_payload_preserved": true, "withdrawn_response_refused": true,
            "prepublication_withdrawal_preserved": true,
            "repeated_sequence_refused": true, "dead_host_watchdog_terminated": true}));
    }
    Ok(
        json!({"status": "passed", "qualified_experts": spec.experts, "rows": rows,
        "owners": devices.iter().map(|device| json!({"name": device.info().name, "selector": device.info().selector})).collect::<Vec<_>>(),
        "exchange_kernels": kernels.iter().map(kernel_identity).collect::<Vec<_>>(),
        "concurrent": concurrent, "lifecycle": lifecycle, "cache": cache.snapshot(),
        "scope": "Untimed dual-GPU activation/contribution handshake with one shared physical CPU cache and fixture readout operands; no serving dispatch, routed activation capture, admission or speed claim"}),
    )
}
