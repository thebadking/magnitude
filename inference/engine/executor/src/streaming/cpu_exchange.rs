//! Single-row activation/contribution publication while a CUDA graph is parked.
//! A channel belongs to one ordered device queue. Failure permanently poisons it;
//! no later launch may overwrite a payload a late host reader can still retain.
use super::words::{SharedBytes, SharedWords};
use seismic::{BackendName, Device, Element, Tensor};
use std::sync::{
    Arc,
    atomic::{AtomicI32, Ordering},
};

const REQUEST: usize = 0;
const ACKNOWLEDGED: usize = 1;
const FAILURE: usize = 2;
const CONSUMED: usize = 3;

pub struct CpuExpertExchange {
    activation: Tensor,
    contribution: Tensor,
    control: Tensor,
    heartbeat: Tensor,
    activation_bytes: SharedBytes,
    contribution_bytes: SharedBytes,
    words: SharedWords,
    beat: SharedWords,
    claimed: AtomicI32,
    ticks: AtomicI32,
    width: usize,
}

/// An activation copied after the device's release publication. Dropping an
/// unanswered request fails the channel. The owner survives the host consumer.
pub struct CpuExpertRequest {
    owner: Arc<CpuExpertExchange>,
    sequence: i32,
    activation: Vec<u8>,
    answered: bool,
}

impl CpuExpertExchange {
    /// The caller's host claim must cover these mapped buffers before allocation.
    /// This forms no numerical program and grants no CPU execution admission.
    pub fn new(device: &Device, width: u64) -> Result<Arc<Self>, String> {
        if device.backend() != BackendName::Cuda || width == 0 {
            return Err("CPU expert exchange requires CUDA and a positive width".into());
        }
        let width = usize::try_from(width).map_err(|_| "exchange width exceeds the host")?;
        width.checked_mul(6).ok_or("exchange size overflow")?;
        let allocate = |element, shape: &[u64]| {
            Tensor::shared(device, element, shape).map_err(|e| e.to_string())
        };
        let activation = allocate(Element::bf16(), &[1, width as u64])?;
        let contribution = allocate(Element::f32(), &[1, width as u64])?;
        let control = allocate(Element::i32(), &[4])?;
        let heartbeat = allocate(Element::i32(), &[1])?;
        let words = SharedWords::new(control.clone())?;
        let beat = SharedWords::new(heartbeat.clone())?;
        words.fill(0);
        beat.fill(0);
        Ok(Arc::new(Self {
            activation_bytes: SharedBytes::new(activation.clone())?,
            contribution_bytes: SharedBytes::new(contribution.clone())?,
            activation,
            contribution,
            control,
            heartbeat,
            words,
            beat,
            claimed: AtomicI32::new(0),
            ticks: AtomicI32::new(0),
            width,
        }))
    }

    pub fn activation(&self) -> &Tensor {
        &self.activation
    }
    pub fn contribution(&self) -> &Tensor {
        &self.contribution
    }
    pub fn control(&self) -> &Tensor {
        &self.control
    }
    pub fn heartbeat(&self) -> &Tensor {
        &self.heartbeat
    }

    /// Call while waiting/servicing, as with the ordinary streaming watcher.
    pub fn pulse(&self) {
        let tick = self.ticks.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        self.beat.store(0, tick);
    }

    pub fn failure(&self) -> Option<i32> {
        match self.words.load(FAILURE) {
            0 => None,
            sequence => Some(sequence),
        }
    }
    pub fn consumed(&self, sequence: i32) -> bool {
        self.words.load(CONSUMED) == sequence
    }

    /// Permanently withdraw a channel, including before a request arrives.
    pub fn abort(&self) {
        let sequence = self.words.load(REQUEST);
        self.words
            .store(FAILURE, if sequence > 0 { sequence } else { -1 });
    }

    /// Exactly one host consumer claims each strictly increasing sequence.
    /// Payload access never uses synchronized tensor readback on the parked queue.
    pub fn try_request(self: &Arc<Self>) -> Result<Option<CpuExpertRequest>, String> {
        if self.failure().is_some() {
            return Err("CPU expert exchange is poisoned".into());
        }
        let sequence = self.words.load(REQUEST);
        let previous = self.claimed.load(Ordering::Acquire);
        if sequence <= previous {
            return Ok(None);
        }
        if self
            .claimed
            .compare_exchange(previous, sequence, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(None);
        }
        let count = self.width * 2;
        let mut activation = vec![0; count];
        // SAFETY: the release request follows every GPU payload store. Until
        // this request answers, the GPU waits. A timeout poisons the channel,
        // so even a queued successor cannot overwrite a late reader's payload.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.activation_bytes.at(0, count as u64),
                activation.as_mut_ptr(),
                count,
            );
        }
        Ok(Some(CpuExpertRequest {
            owner: self.clone(),
            sequence,
            activation,
            answered: false,
        }))
    }
}

impl CpuExpertRequest {
    pub fn sequence(&self) -> i32 {
        self.sequence
    }
    pub fn activation(&self) -> &[u8] {
        &self.activation
    }

    /// Supply the expert's activation-rounded F32 contribution, before router
    /// weighting. CPU work and any read lease must have physically completed.
    pub fn complete(mut self, contribution: &[u8]) -> Result<(), String> {
        if contribution.len() != self.owner.width * 4 {
            return Err("CPU expert contribution has the wrong byte length".into());
        }
        if self.owner.failure().is_some() || self.owner.words.load(REQUEST) != self.sequence {
            return Err("CPU expert request was withdrawn".into());
        }
        // SAFETY: the sole consumer owns this response. The GPU cannot read it
        // before ACK; after failure it never reads it or reuses the channel.
        unsafe {
            std::ptr::copy_nonoverlapping(
                contribution.as_ptr(),
                self.owner
                    .contribution_bytes
                    .at(0, contribution.len() as u64),
                contribution.len(),
            );
        }
        if self.owner.failure().is_some() {
            return Err("CPU expert request failed before response publication".into());
        }
        self.answered = true;
        self.owner.words.store(ACKNOWLEDGED, self.sequence);
        Ok(())
    }
}

impl Drop for CpuExpertRequest {
    fn drop(&mut self) {
        if !self.answered {
            self.owner.words.store(FAILURE, self.sequence);
        }
    }
}
