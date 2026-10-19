//! The worker binds one program family before constructing its numerical domain.
//! Every lane keeps its concrete submission type through completion and finish.

use crate::programs::{
    CompletedHeadWork, CompletedStateWork, CompletedTargetWork, CompletedVisionWork, HeadProgram,
    ProgramSubmission, StateProgram, SubmittedHead, SubmittedTarget, TargetProgram, VisionProgram,
};
use crate::{
    AttestedPrograms, ResidentHead, ResidentTarget, ResidentVision, SubmitError,
    ValidatedHeadLaunch, ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch,
};
use magnitude_family_contracts::{Decoder, ModelDefinition};
use std::rc::Rc;

pub trait ProgramFamily: 'static {
    type TargetSubmission: ProgramSubmission<CompletedWork = CompletedTargetWork>
        + SubmittedTarget
        + 'static;
    type HeadSubmission: ProgramSubmission<CompletedWork = CompletedHeadWork>
        + SubmittedHead
        + 'static;
    type VisionSubmission: ProgramSubmission<CompletedWork = CompletedVisionWork> + 'static;
    type StateSubmission: ProgramSubmission<CompletedWork = CompletedStateWork> + 'static;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String>;
    fn head_is_bound(&self) -> bool;
    fn vision_is_bound(&self) -> bool;
    fn state_is_bound(&self) -> bool;
    fn unbind_optional(&mut self);

    /// Physical target tensors held by the bound family. Test families with
    /// no resident model have no target charge.
    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn peer_weight_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }
    fn target_bank_source_bytes(&self) -> u64 {
        0
    }
    fn peer_bank_source_bytes(&self) -> u64 {
        0
    }
    fn host_expert_cache_bytes(&self) -> u64 {
        0
    }
    fn reclaim_host_expert_cache(&self) {}
    fn peer_program_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }
    fn peer_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    /// Model/program holdings of the second physical owner, when present.
    fn peer_model_bytes(&self) -> Result<u64, &'static str> {
        let weights = self.peer_weight_bytes()?;
        let programs = self.peer_program_bytes()?;
        let constants = self.peer_constant_bytes()?;
        weights
            .checked_add(programs)
            .and_then(|bytes| bytes.checked_add(constants))
            .ok_or("peer model holdings overflow")
    }

    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn target_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        Ok(0)
    }

    fn bound_constant_bytes(&self) -> Result<u64, &'static str> {
        self.target_constant_bytes()?
            .checked_add(self.optional_constant_bytes()?)
            .ok_or("bound graph constant charge overflows")
    }

    /// Submit a chain of target launches (`TargetProgram::submit`).
    fn submit_target(
        &mut self,
        chain: Vec<ValidatedTargetLaunch>,
    ) -> Result<Vec<Self::TargetSubmission>, (SubmitError, Vec<ValidatedTargetLaunch>)>;
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)>;
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)>;
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)>;
}

pub struct NativeFamily {
    peer: Option<(
        Box<NativeFamily>,
        Rc<std::cell::RefCell<crate::AllocatedResources>>,
    )>,
    programs: Rc<AttestedPrograms>,
    resident: ResidentTarget,
    target: crate::programs::native_target::NativeTargetProgram,
    state: crate::programs::native_state::NativeStateProgram,
    head: Option<crate::programs::native_drafter::NativeDrafterProgram>,
    vision: Option<crate::programs::native_vision::NativeVisionProgram>,
}

impl NativeFamily {
    pub(crate) fn new_placed(
        programs: Rc<AttestedPrograms>,
        resident: ResidentTarget,
        peer_programs: Rc<AttestedPrograms>,
        peer_resident: ResidentTarget,
        peer_resources: Rc<std::cell::RefCell<crate::AllocatedResources>>,
        geometry: Decoder,
    ) -> Result<Self, String> {
        let first = programs
            .target_graphs()
            .ok_or("first owner lacks target graphs")?
            .owned_blocks();
        let second = peer_programs
            .target_graphs()
            .ok_or("second owner lacks target graphs")?
            .owned_blocks();
        if first.start != 0
            || first.end != second.start
            || second.end != geometry.blocks.len()
            || resident
                .embedding
                .tensor()
                .device()
                .same_device(&peer_resident.embedding.tensor().device())
        {
            return Err("native target owners do not cover the decoder on two devices".into());
        }
        let mut family = Self::new(programs, resident, geometry.clone())?;
        let peer = Self::new(peer_programs, peer_resident, geometry)?;
        family.peer = Some((Box::new(peer), peer_resources));
        Ok(family)
    }

    pub(crate) fn new(
        programs: Rc<AttestedPrograms>,
        resident: ResidentTarget,
        geometry: Decoder,
    ) -> Result<Self, String> {
        let binding_constants = programs
            .target_graphs()
            .ok_or("target graph family was not prepared")?
            .binding_constant_bytes()?;
        let target = programs
            .bind_target(resident.clone(), geometry)
            .map_err(|error| error.to_string())?;
        if target.constant_bytes()? != binding_constants {
            return Err("bound target graph constants differ from the claimed charge".into());
        }
        let state = programs.bind_state();
        Ok(Self {
            peer: None,
            programs,
            resident,
            target,
            state,
            head: None,
            vision: None,
        })
    }
}

impl ProgramFamily for NativeFamily {
    type TargetSubmission = crate::programs::native_target::NativeTargetSubmission;
    type HeadSubmission =
        <crate::programs::native_drafter::NativeDrafterProgram as HeadProgram>::Submission;
    type VisionSubmission =
        <crate::programs::native_vision::NativeVisionProgram as VisionProgram>::Submission;
    type StateSubmission =
        <crate::programs::native_state::NativeStateProgram as StateProgram>::Submission;

    fn bind_head(
        &mut self,
        resident: ResidentHead,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.peer.is_some() {
            return Err("distributed target does not yet support drafting".into());
        }
        if self.head.is_none() {
            let binding_constants = self
                .programs
                .drafter_graphs()
                .ok_or("head graph family was not prepared")?
                .binding_constant_bytes()?;
            let head = self
                .programs
                .bind_head(resident, definition)
                .map_err(|error| error.to_string())?;
            if head.constant_bytes()? != binding_constants {
                return Err("bound head graph constants differ from the claimed charge".into());
            }
            self.head = Some(head);
        }
        Ok(())
    }
    fn bind_vision(
        &mut self,
        resident: ResidentVision,
        definition: &ModelDefinition,
    ) -> Result<(), String> {
        if self.peer.is_some() {
            return Err("distributed target does not yet support vision".into());
        }
        if self.vision.is_none() {
            self.vision = Some(
                self.programs
                    .bind_vision(resident, definition)
                    .map_err(|error| error.to_string())?,
            );
        }
        Ok(())
    }
    fn head_is_bound(&self) -> bool {
        self.head.is_some()
    }
    fn vision_is_bound(&self) -> bool {
        self.vision.is_some()
    }
    fn state_is_bound(&self) -> bool {
        true
    }

    fn unbind_optional(&mut self) {
        self.head = None;
        self.vision = None;
    }

    fn target_weight_bytes(&self) -> Result<u64, &'static str> {
        self.resident.storage_bytes()
    }

    fn target_bank_source_bytes(&self) -> u64 {
        self.resident
            .bank
            .as_ref()
            .map_or(0, |bank| bank.source_control_bytes)
    }

    fn host_expert_cache_bytes(&self) -> u64 {
        self.resident.bank.as_ref().map_or(0, |bank| {
            bank.host_experts
                .as_ref()
                .map_or(0, |cache| cache.allocated_bytes())
                .saturating_add(
                    bank.cpu_experts
                        .as_ref()
                        .map_or(0, |service| service.allocated_bytes()),
                )
        })
    }

    fn reclaim_host_expert_cache(&self) {
        if let Some(service) = self
            .resident
            .bank
            .as_ref()
            .and_then(|bank| bank.cpu_experts.as_ref())
        {
            service.reclaim_if_needed();
        }
        if let Some(cache) = self
            .resident
            .bank
            .as_ref()
            .and_then(|bank| bank.host_experts.as_ref())
        {
            cache.reclaim_if_needed();
        }
    }

    fn peer_bank_source_bytes(&self) -> u64 {
        self.peer
            .as_ref()
            .map_or(0, |(peer, _)| peer.target_bank_source_bytes())
    }

    fn peer_weight_bytes(&self) -> Result<u64, &'static str> {
        self.peer
            .as_ref()
            .map_or(Ok(0), |(peer, _)| peer.target_weight_bytes())
    }
    fn peer_program_bytes(&self) -> Result<u64, &'static str> {
        self.peer
            .as_ref()
            .map_or(Ok(0), |(peer, _)| peer.prepared_program_bytes())
    }
    fn peer_constant_bytes(&self) -> Result<u64, &'static str> {
        self.peer
            .as_ref()
            .map_or(Ok(0), |(peer, _)| peer.target_constant_bytes())
    }

    fn prepared_program_bytes(&self) -> Result<u64, &'static str> {
        self.programs.device_storage_bytes()
    }

    fn target_constant_bytes(&self) -> Result<u64, &'static str> {
        self.target.constant_bytes()
    }

    fn optional_constant_bytes(&self) -> Result<u64, &'static str> {
        let head = self
            .head
            .as_ref()
            .map(|head| head.constant_bytes())
            .transpose()?
            .unwrap_or(0);
        let vision = self
            .vision
            .as_ref()
            .map(|vision| vision.constant_bytes())
            .transpose()?
            .unwrap_or(0);
        head.checked_add(vision)
            .ok_or("optional graph constant charge overflows")
    }

    fn submit_target(
        &mut self,
        chain: Vec<ValidatedTargetLaunch>,
    ) -> Result<Vec<Self::TargetSubmission>, (SubmitError, Vec<ValidatedTargetLaunch>)> {
        use crate::programs::native_target::NativeTargetSubmission;
        if let Some((peer, resources)) = &self.peer {
            let reserved = (0..chain.len())
                .map(|_| resources.borrow().reserve_target_range())
                .collect::<Result<Vec<_>, _>>();
            let reserved = match reserved {
                Ok(reserved) => reserved,
                Err(error) => {
                    return Err((
                        SubmitError::Invariant(crate::InvariantError {
                            context: "distributed target reservation",
                            detail: error.to_string(),
                        }),
                        chain,
                    ));
                }
            };
            self.target
                .submit_pair(&peer.target, chain, reserved)
                .map(|submissions| {
                    submissions
                        .into_iter()
                        .map(NativeTargetSubmission::placed)
                        .collect()
                })
        } else {
            self.target.submit(chain).map(|submissions| {
                submissions
                    .into_iter()
                    .map(NativeTargetSubmission::single)
                    .collect()
            })
        }
    }
    fn submit_head(
        &mut self,
        launch: ValidatedHeadLaunch,
    ) -> Result<Self::HeadSubmission, (SubmitError, ValidatedHeadLaunch)> {
        self.head
            .as_mut()
            .expect("head was bound before submission")
            .submit(launch)
    }
    fn submit_vision(
        &mut self,
        launch: ValidatedVisionLaunch,
    ) -> Result<Self::VisionSubmission, (SubmitError, ValidatedVisionLaunch)> {
        self.vision
            .as_mut()
            .expect("vision was bound before submission")
            .submit(launch)
    }
    fn submit_state(
        &mut self,
        launch: ValidatedStateLaunch,
    ) -> Result<Self::StateSubmission, (SubmitError, ValidatedStateLaunch)> {
        if let Some((peer, resources)) = &self.peer {
            let workspace = match resources.borrow().state_graph().acquire_workspace() {
                Ok(workspace) => workspace,
                Err(error) => {
                    return Err((
                        SubmitError::Invariant(crate::InvariantError {
                            context: "distributed state reservation",
                            detail: error.to_string(),
                        }),
                        launch,
                    ));
                }
            };
            self.state.submit_pair(&peer.state, workspace, launch)
        } else {
            self.state.submit(launch)
        }
    }
}
