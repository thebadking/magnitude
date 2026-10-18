//! Recurrent components grouped by their device owner. Every group backs the
//! same bank indices; growth publishes those indices only after all groups have
//! allocated them. Logical component order is independent of physical grouping.
use seismic::{Device, SlabRegion, SlabTensor, Tensor, TensorError, TensorStorageObserver};
use std::rc::Rc;

pub(crate) struct BankStorage {
    groups: Vec<(Rc<Device>, SlabTensor)>,
    regions: Vec<(usize, usize)>,
}

impl BankStorage {
    pub fn new(
        devices: &[Rc<Device>],
        rows_per_slab: u64,
        logical_rows: u64,
        regions: Vec<SlabRegion>,
    ) -> Result<Self, TensorError> {
        if regions.is_empty() || regions.len() != devices.len() {
            return Err(TensorError::SlabLayout(
                "bank component owners differ from regions".into(),
            ));
        }
        let mut grouped: Vec<(Rc<Device>, Vec<SlabRegion>)> = Vec::new();
        let mut mapping = Vec::new();
        for (device, region) in devices.iter().zip(regions) {
            let group = match grouped
                .iter()
                .position(|(owner, _)| owner.same_device(device))
            {
                Some(group) => group,
                None => {
                    grouped.push((device.clone(), Vec::new()));
                    grouped.len() - 1
                }
            };
            mapping.push((group, grouped[group].1.len()));
            grouped[group].1.push(region);
        }
        let groups = grouped
            .into_iter()
            .map(|(device, regions)| {
                SlabTensor::new(&device, rows_per_slab, logical_rows, regions)
                    .map(|slabs| (device, slabs))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            groups,
            regions: mapping,
        })
    }

    pub fn groups(&self) -> impl Iterator<Item = &SlabTensor> {
        self.groups.iter().map(|(_, slabs)| slabs)
    }

    pub fn storage_bytes(&self) -> u64 {
        self.groups().map(SlabTensor::storage_bytes).sum()
    }

    pub fn storage_bytes_on(&self, device: &Device) -> u64 {
        self.groups
            .iter()
            .filter(|(owner, _)| device.same_device(owner))
            .map(|(_, slabs)| slabs.storage_bytes())
            .sum()
    }

    pub fn slab_bytes(&self) -> u64 {
        self.slab_bytes_on(None)
    }

    pub fn slab_bytes_on(&self, device: Option<&Device>) -> u64 {
        self.groups
            .iter()
            .filter(|(owner, _)| device.is_none_or(|device| device.same_device(owner)))
            .map(|(_, slabs)| slabs.slab_bytes())
            .sum()
    }

    pub fn slab_storage_observers(&self) -> impl Iterator<Item = TensorStorageObserver> + '_ {
        self.groups().flat_map(SlabTensor::slab_storage_observers)
    }

    pub fn logical_region(&self, region: usize) -> Result<Tensor, TensorError> {
        let &(group, region) = self
            .regions
            .get(region)
            .ok_or_else(|| TensorError::SlabLayout("unknown bank component".into()))?;
        self.groups[group].1.logical_region(region)
    }

    /// All groups have the same backed indices. The representative slabs are
    /// used only for bank placement, never to enumerate physical charges.
    pub fn slabs(&self) -> impl Iterator<Item = (usize, &seismic::Slab)> {
        self.groups[0].1.slabs()
    }

    #[cfg(test)]
    pub fn slab(&self, index: usize) -> Option<&seismic::Slab> {
        self.groups[0].1.slab(index)
    }

    pub fn add_slab(&mut self) -> Result<usize, TensorError> {
        let mut added = Vec::new();
        for group in 0..self.groups.len() {
            match self.groups[group].1.add_slab() {
                Ok(index) => {
                    assert!(added.first().is_none_or(|&previous| previous == index));
                    added.push(index);
                }
                Err(error) => {
                    if let Some(&index) = added.first() {
                        let mut stores = self.groups[..added.len()]
                            .iter_mut()
                            .map(|(_, slabs)| slabs)
                            .collect::<Vec<_>>();
                        SlabTensor::free_slab_group(&mut stores, index)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(added[0])
    }

    pub fn free_slab(
        &mut self,
        index: usize,
    ) -> Result<Option<Vec<TensorStorageObserver>>, TensorError> {
        let mut stores = self
            .groups
            .iter_mut()
            .map(|(_, slabs)| slabs)
            .collect::<Vec<_>>();
        let observers = SlabTensor::free_slab_group(&mut stores, index)?;
        Ok((!observers.is_empty()).then_some(observers))
    }
}
