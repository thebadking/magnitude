//! Canonical, untuned CPU companion programs from the immutable bank plan.
//! Search collects their formation in the load's existing first-load allowance.
use crate::planning::{CpuExpertBinding, CpuExpertPlan};
use crate::{DEFAULT_KERNEL_CACHE_BYTES, KernelCache};
use magnitude_kernels::{repack_weight, routed_cpu_expert};
use seismic::{
    BoundEntry, Device, DeviceCatalog, DeviceOptions, DeviceSelector, Entry, NativeKernel,
    NativeSpecialization,
};
use std::sync::Arc;

pub struct CpuExpertPrograms {
    pub plan: Arc<CpuExpertPlan>,
    pub device: Device,
    pub entries: Vec<Arc<CpuExpertEntry>>,
}
impl std::fmt::Debug for CpuExpertPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuExpertPrograms")
            .field("plan", &self.plan)
            .field("device", &self.device.info().selector)
            .finish_non_exhaustive()
    }
}

pub struct CpuExpertEntry {
    pub binding: CpuExpertBinding,
    pub import: [NativeKernel<repack_weight::Entry>; 3],
    pub contribution: NativeKernel<routed_cpu_expert::Entry>,
}

pub(crate) struct CpuExpertProgramContext {
    pub plan: Arc<CpuExpertPlan>,
    pub device: Device,
}

impl CpuExpertProgramContext {
    pub fn open(plan: Arc<CpuExpertPlan>, cache: Option<&KernelCache>) -> Result<Self, String> {
        let catalog = DeviceCatalog::discover().map_err(|e| e.to_string())?;
        let artifacts = cache
            .map(|cache| {
                KernelCache::open(cache.root().to_owned(), DEFAULT_KERNEL_CACHE_BYTES)
                    .map(|cache| Arc::new(cache) as Arc<dyn seismic::ArtifactStore>)
                    .map_err(|e| e.to_string())
            })
            .transpose()?;
        let device = catalog
            .open_with(
                catalog
                    .resolve(DeviceSelector::HostCpu)
                    .map_err(|e| e.to_string())?,
                DeviceOptions {
                    artifacts,
                    ..DeviceOptions::default()
                },
            )
            .map_err(|e| e.to_string())?;
        Ok(Self { plan, device })
    }
    pub fn imports(binding: &CpuExpertBinding) -> [BoundEntry<repack_weight::Entry>; 3] {
        [0, 1, 2].map(|kind| {
            repack_weight::native_entry_with(repack_weight::Elements {
                E: binding.source[kind],
                U: binding.resident[kind],
            })
        })
    }
    pub fn contribution(binding: &CpuExpertBinding) -> BoundEntry<routed_cpu_expert::Entry> {
        routed_cpu_expert::native_entry_with(routed_cpu_expert::Elements {
            EGW: binding.resident[0],
            EUW: binding.resident[1],
            EDW: binding.resident[2],
        })
    }
    pub fn specialization<E: Entry>(
        &self,
        entry: &BoundEntry<E>,
        statics: NativeSpecialization,
    ) -> Result<NativeSpecialization, String> {
        entry
            .default_specialization(&self.device, &statics)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("{} has no canonical CPU implementation", E::NAME))
    }
    pub fn form(self) -> Result<Arc<CpuExpertPrograms>, String> {
        let mut entries = Vec::new();
        let mut shared_imports: std::collections::HashMap<
            (seismic::Element, seismic::Element),
            NativeKernel<repack_weight::Entry>,
        > = std::collections::HashMap::new();
        for binding in &self.plan.bindings {
            let imports = Self::imports(binding)
                .into_iter()
                .enumerate()
                .map(|(kind, entry)| {
                    let key = (binding.source[kind], binding.resident[kind]);
                    if let Some(ready) = shared_imports.get(&key) {
                        return Ok(ready.clone());
                    }
                    let kernel = entry
                        .prepare(
                            &self.device,
                            &self.specialization(&entry, NativeSpecialization::new())?,
                        )
                        .map_err(|e| e.to_string())?;
                    shared_imports.insert(key, kernel.clone());
                    Ok::<_, String>(kernel)
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| "CPU import triple")?;
            let entry = Self::contribution(binding);
            let specialization = self.specialization(
                &entry,
                NativeSpecialization::new()
                    .with_static("H", binding.hidden())
                    .with_static("F", binding.features()),
            )?;
            let contribution = entry
                .prepare(&self.device, &specialization)
                .map_err(|e| e.to_string())?;
            entries.push(Arc::new(CpuExpertEntry {
                binding: binding.clone(),
                import: imports,
                contribution,
            }));
        }
        Ok(Arc::new(CpuExpertPrograms {
            plan: self.plan,
            device: self.device,
            entries,
        }))
    }
}
