from .fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler


def get_sample_scheduler(num_train_timesteps, sampling_steps, sample_solver, device, shift=5.0):
    """
    Set sample scheduler.

        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'uni_pc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 50):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
    """
    sampling_sigmas = None
    if sample_solver == "uni_pc":
        # here
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
    elif sample_solver == "dpm++":
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
    else:
        raise NotImplementedError(f"Unsupported solver type {sample_solver}.")
    return sample_scheduler, sampling_sigmas, timesteps


__all__ = ["get_sample_scheduler"]
