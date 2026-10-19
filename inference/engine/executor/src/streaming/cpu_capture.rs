//! Bounded observations of actual CPU dispatches; never numerical admission.
use seismic::Tensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    directory: PathBuf,
    first_sequence: i32,
    last_sequence: i32,
    max_records: usize,
}

pub(super) struct CpuExpertCapture {
    configuration: Configuration,
    records: AtomicUsize,
}

pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

impl CpuExpertCapture {
    pub fn requested() -> Result<Option<Self>, String> {
        let Some(value) = std::env::var_os("MAGNITUDE_CAPTURE_CPU_EXPERTS") else {
            return Ok(None);
        };
        let value = value.to_str().ok_or("invalid CPU capture configuration")?;
        if value.len() > 4096 {
            return Err("CPU capture configuration exceeds 4096 bytes".into());
        }
        let configuration: Configuration =
            serde_json::from_str(value).map_err(|e| e.to_string())?;
        if configuration.first_sequence <= 0
            || configuration.last_sequence < configuration.first_sequence
            || configuration.last_sequence - configuration.first_sequence > 128
            || configuration.max_records == 0
            || configuration.max_records > 128
        {
            return Err(
                "CPU capture requires at most 128 records over at most 129 positive sequences"
                    .into(),
            );
        }
        std::fs::create_dir_all(&configuration.directory).map_err(|e| e.to_string())?;
        Ok(Some(Self {
            configuration,
            records: AtomicUsize::new(0),
        }))
    }
    pub fn owned_bytes(&self) -> u64 {
        (std::mem::size_of::<Self>() + self.configuration.directory.capacity()) as u64
    }
    pub fn select(&self, sequence: i32) -> bool {
        (self.configuration.first_sequence..=self.configuration.last_sequence).contains(&sequence)
            && self
                .records
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |records| {
                    (records < self.configuration.max_records).then_some(records + 1)
                })
                .is_ok()
    }
    fn stem(sequence: i32, layer: usize, expert: usize) -> String {
        format!("{sequence:08}-{layer:02}-{expert:04}")
    }
    pub fn record(
        &self,
        sequence: i32,
        layer: usize,
        expert: usize,
        source: &str,
        weights: &[Tensor; 3],
        input: &[u8],
        output: &[u8],
    ) -> Result<(), String> {
        let stem = Self::stem(sequence, layer, expert);
        let mut hashes = Vec::new();
        for weight in weights {
            let bytes = weight.read_to_host().map_err(|e| e.to_string())?;
            hashes.push(digest(&bytes));
        }
        let input_name = format!("{stem}.bf16");
        let output_name = format!("{stem}.f32");
        let path = &self.configuration.directory;
        std::fs::write(path.join(&input_name), input).map_err(|e| e.to_string())?;
        std::fs::write(path.join(&output_name), output).map_err(|e| e.to_string())?;
        let sources: serde_json::Value = serde_json::from_str(source).map_err(|e| e.to_string())?;
        let record = serde_json::json!({"schema":1,"sequence":sequence,"layer":layer,"expert":expert,
            "sources":sources,"elements":weights.each_ref().map(|w|w.element().name()),
            "cached_weight_sha256":hashes,"input":input_name,"input_sha256":digest(input),
            "output":output_name,"output_sha256":digest(output),"hidden":input.len()/2,
            "meaning":"Actual unweighted CPU expert contribution before GPU publication; consumption marker is separate"});
        std::fs::write(
            path.join(format!("{stem}.json")),
            serde_json::to_vec(&record).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
    pub fn consumed(&self, sequence: i32, layer: usize, expert: usize) -> Result<(), String> {
        let stem = Self::stem(sequence, layer, expert);
        if self
            .configuration
            .directory
            .join(format!("{stem}.json"))
            .is_file()
        {
            std::fs::write(
                self.configuration
                    .directory
                    .join(format!("{stem}.consumed")),
                b"GPU consumed\n",
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn failed(&self, error: &str) {
        eprintln!("CPU expert capture failed: {error}");
        let _ = std::fs::write(self.configuration.directory.join("failure.txt"), error);
    }
}
