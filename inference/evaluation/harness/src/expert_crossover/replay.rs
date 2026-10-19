//! Replay actual fixture-produced CPU dispatch operands against fresh imports.
use super::*;

#[derive(Args)]
pub struct ReplayOptions {
    #[arg(long)]
    capture: PathBuf,
    #[arg(long)]
    gpu_selector: String,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Evaluate an untimed scalar group-factored CPU recipe; never serving.
    #[arg(long)]
    grouped_cpu_recipe: bool,
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("missing capture {key}"))
}
fn natural(value: &Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("missing capture {key}"))
}
fn payload(
    root: &std::path::Path,
    record: &Value,
    key: &str,
    bytes: u64,
) -> Result<Vec<u8>, String> {
    let name = string(record, key)?;
    require(
        std::path::Path::new(name)
            .file_name()
            .and_then(|n| n.to_str())
            == Some(name),
        "capture payload must be a local file name",
    )?;
    let path = root.join(name);
    require(
        std::fs::metadata(&path).map_err(why)?.len() == bytes,
        "capture payload size differs",
    )?;
    let data = std::fs::read(path).map_err(why)?;
    require(
        record[format!("{key}_sha256")] == sha256_hex(&data),
        "capture payload digest differs",
    )?;
    Ok(data)
}

fn weights(inputs: &Inputs, record: &Value) -> Result<[Weight; 3], String> {
    let host = crate::host(&inputs.artifacts)?;
    let sources = record["sources"]
        .as_array()
        .ok_or("missing captured source inventory")?;
    require(sources.len() == 3, "capture source triple differs")?;
    let expert = natural(record, "expert")?;
    let mut result = Vec::new();
    for source_id in sources {
        let name = string(source_id, "tensor")?;
        let stored = host.package().target().tensor(name).map_err(why)?;
        require(
            stored.shape.len() == 3 && expert < stored.shape[0],
            "captured expert is outside the source",
        )?;
        require(
            source_id["path"] == json!(stored.source.path())
                && source_id["offset"] == stored.offset
                && source_id["bytes"] == stored.nbytes,
            "capture source range differs from the catalog artifact",
        )?;
        let (source, cpu, gpu) = representations(stored.encoding)?;
        require(
            source_id["source"] == source.name()
                && source_id["matrix"] == json!([stored.shape[1], stored.shape[2]]),
            "capture source representation differs",
        )?;
        let shape = vec![1, stored.shape[1], stored.shape[2]];
        let record_bytes = source.canonical_byte_len(&shape).map_err(why)?;
        require(
            record_bytes <= 64 << 20
                && record_bytes.checked_mul(stored.shape[0]) == Some(stored.nbytes),
            "capture source geometry exceeds its bound",
        )?;
        let offset = stored
            .offset
            .checked_add(
                expert
                    .checked_mul(record_bytes)
                    .ok_or("expert offset overflow")?,
            )
            .ok_or("source offset overflow")?;
        let raw = stored
            .source
            .read(offset, record_bytes as usize)
            .map_err(why)?;
        result.push(Weight {
            shape,
            raw,
            source,
            cpu,
            gpu,
            record_bytes: record_bytes as usize,
            identity: source_id.clone(),
        });
    }
    let result = triple(result)?;
    require(
        result[0].shape == result[1].shape
            && result[2].shape == [1, result[0].shape[2], result[0].shape[1]],
        "captured matrix relationship differs",
    )?;
    Ok(result)
}

pub fn run(
    inputs: &Inputs,
    options: &ReplayOptions,
    digest: &str,
) -> Result<(Value, Vec<String>), String> {
    let producer: Value =
        serde_json::from_slice(&std::fs::read(options.capture.join("producer.json")).map_err(why)?)
            .map_err(why)?;
    require(
        producer["tool"] == "bench.cpu_expert_capture"
            && producer["model"] == inputs.artifacts.artifacts.model
            && producer["target_sha256"] == inputs.artifacts.artifacts.target_sha256,
        "capture producer or model differs",
    )?;
    require(
        producer["fixture_set_digest"] == inputs.set.digest
            && producer["rendered_digest"] == digest,
        "actual routed capture is not bound to this rendered fixture",
    )?;
    let rendered = RenderedFile::read(
        &options.capture.join("rendered.jsonl"),
        &inputs.set.digest,
        &inputs.artifacts.artifacts.model,
        &inputs.artifacts.artifacts.target_sha256,
    )?;
    require(
        rendered.digest == digest
            && producer["canonical_parent_entries"] == true
            && producer["sequence_range_verified"] == true,
        "capture request range or canonical arithmetic was not verified",
    )?;
    require(
        !options.capture.join("failure.txt").exists(),
        "serving capture reported a failure",
    )?;
    let selector: DeviceSelector = serde_json::from_str(&options.gpu_selector).map_err(why)?;
    require(
        matches!(selector, DeviceSelector::Cuda { .. }),
        "replay requires an exact CUDA selector",
    )?;
    let artifacts: Option<Arc<dyn seismic::ArtifactStore>> = options
        .cache_dir
        .as_ref()
        .map(|path| {
            magnitude_executor::KernelCache::open(
                path.clone(),
                magnitude_executor::DEFAULT_KERNEL_CACHE_BYTES,
            )
            .map(|cache| Arc::new(cache) as Arc<dyn seismic::ArtifactStore>)
            .map_err(why)
        })
        .transpose()?;
    let catalog = DeviceCatalog::discover().map_err(why)?;
    let device_options = DeviceOptions {
        artifacts,
        ..DeviceOptions::default()
    };
    let cpu_device = catalog
        .open_with(
            catalog.resolve(DeviceSelector::HostCpu).map_err(why)?,
            device_options.clone(),
        )
        .map_err(why)?;
    let gpu_device = catalog
        .open_with(catalog.resolve(selector).map_err(why)?, device_options)
        .map_err(why)?;
    cpu_device.set_memory_limit(Some(1 << 30));
    gpu_device.set_memory_limit(Some(1 << 30));
    let activations = options
        .grouped_cpu_recipe
        .then(|| super::grouped::Activations::new(&gpu_device))
        .transpose()?;
    let mut paths = std::fs::read_dir(&options.capture)
        .map_err(why)?
        .map(|entry| entry.map(|e| e.path()).map_err(why))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.extension().is_some_and(|e| e == "json")
            && path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.as_bytes().first().is_some_and(u8::is_ascii_digit))
    });
    paths.sort();
    require(
        !paths.is_empty() && paths.len() <= 128,
        "replay needs 1..128 actual captures",
    )?;
    let mut rows = Vec::new();
    for path in paths {
        if !path.with_extension("consumed").is_file() {
            continue;
        }
        require(
            std::fs::metadata(&path).map_err(why)?.len() <= 32768,
            "capture metadata exceeds 32 KiB",
        )?;
        let record: Value =
            serde_json::from_slice(&std::fs::read(&path).map_err(why)?).map_err(why)?;
        let sequence = natural(&record, "sequence")?;
        require(
            sequence >= natural(&producer, "first_sequence")?
                && sequence <= natural(&producer, "last_sequence")?,
            "captured sequence is outside the fixture request",
        )?;
        let weights = weights(inputs, &record)?;
        let (h, f) = (weights[0].shape[2], weights[0].shape[1]);
        require(record["hidden"] == h, "capture hidden width differs")?;
        let input = payload(&options.capture, &record, "input", h * 2)?;
        let actual = payload(&options.capture, &record, "output", h * 4)?;
        let cpu = Backend::new(cpu_device.clone(), &weights, &[input.clone()], 1)?;
        let gpu = Backend::new(gpu_device.clone(), &weights, &[input.clone()], 1)?;
        let mut native_equal = Vec::new();
        for kind in 0..3 {
            let bytes = cpu.resident[kind].read_to_host().map_err(why)?;
            require(
                record["elements"][kind] == weights[kind].cpu.name(),
                "captured CPU representation differs",
            )?;
            native_equal.push(record["cached_weight_sha256"][kind] == sha256_hex(&bytes));
        }
        let expected_cpu = cache::contribution(&cpu, &cpu.resident, 0, 0, h, f)?;
        let expected_gpu = cache::contribution(&gpu, &gpu.resident, 0, 0, h, f)?;
        let actual =
            Tensor::from_host(&cpu.device, Element::f32(), &[1, h], &actual).map_err(why)?;
        let gpu_product_cpu = Tensor::from_host(
            &cpu.device,
            Element::bf16(),
            &[1, 1, f],
            &expected_gpu.product.read_to_host().map_err(why)?,
        )
        .map_err(why)?;
        let base = Tensor::zeros(&cpu.device, Element::f32(), &[1, h]).map_err(why)?;
        let routes =
            Tensor::from_host(&cpu.device, Element::i32(), &[1, 1], &ints([0])).map_err(why)?;
        let scores = Tensor::from_host(&cpu.device, Element::f32(), &[1, 1], &1_f32.to_le_bytes())
            .map_err(why)?;
        let same_product_cpu = cpu
            .down
            .call(routed_down::Args {
                base: &base,
                product: &gpu_product_cpu,
                routes: &routes,
                weights: &scores,
                expert_down: &cpu.resident[2],
            })
            .map_err(why)?
            .value;
        let grouped = activations.as_ref().map(|activation| {
            let candidate = super::grouped::evaluate(&cpu,&weights,&input,&expected_gpu.product,activation)?;
            Ok::<_,String>(json!({"cpu_silu_product":difference(&expected_gpu.product,&candidate.product)?,
                "cpu_silu_contribution":difference(&expected_gpu.result,&candidate.result)?,
                "gpu_silu_product":difference(&expected_gpu.product,&candidate.product_gpu_silu)?,
                "gpu_silu_contribution":difference(&expected_gpu.result,&candidate.result_gpu_silu)?,
                "down_same_gpu_product":difference(&expected_gpu.result,&candidate.result_same_gpu_product)?,
                "scope":"Scalar grouped coefficients, F64 group partials and F32 folds, one part; no serving or speed claim"}))
        }).transpose()?;
        rows.push(json!({"capture":path.file_name().and_then(|s|s.to_str()),"sequence":sequence,"layer":record["layer"],"expert":record["expert"],"grouped_cpu_recipe":grouped,
            "retained_native_equal_fresh_import":native_equal,
            "serving_cpu_vs_fresh_cpu":difference(&actual,&expected_cpu.result)?,
            "cpu_vs_gpu_product":difference(&expected_gpu.product,&expected_cpu.product)?,
            "cpu_vs_gpu_contribution":difference(&expected_gpu.result,&expected_cpu.result)?,
            "cpu_vs_gpu_down_same_gpu_product":difference(&expected_gpu.result,&same_product_cpu)?}));
    }
    require(
        !rows.is_empty(),
        "no captured contribution was acknowledged by the GPU",
    )?;
    let report = json!({"status":"complete","captured_consumed_experts":rows.len(),
        "all_retained_native_equal_fresh_import":rows.iter().all(|r|r["retained_native_equal_fresh_import"].as_array().is_some_and(|a|a.iter().all(|v|v==&Value::Bool(true)))),
        "all_serving_cpu_equal_fresh_cpu":rows.iter().all(|r|r["serving_cpu_vs_fresh_cpu"]["byte_equal"]==true),
        "rows":rows,"producer":producer,"activation_diagnostic":activations.as_ref().map(|a|&a.diagnostic),
        "scope":"Actual fixture-derived routed operands; canonical one-expert CPU/GPU diagnostic. No numerical admission, model-quality or speed claim"});
    std::fs::write(
        inputs.run.path("expert-replay.json"),
        serde_json::to_vec_pretty(&report).map_err(why)?,
    )
    .map_err(why)?;
    Ok((
        report,
        vec![
            format!("{:?}", cpu_device.info().selector),
            format!("{:?}", gpu_device.info().selector),
        ],
    ))
}
