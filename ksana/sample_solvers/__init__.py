import torch

from .fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .fm_solvers_euler import EulerScheduler


SUPPORTED_SOLVERS = ["uni_pc", "dpm++", "euler"]


def get_sample_scheduler(
    num_train_timesteps, sampling_steps, sample_solver, device, shift=5.0, denoise=1.0, sigmas=None
):
    """
    Set sample scheduler.

        shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter. Affects temporal dynamics
        sample_solver (`str`, *optional*, defaults to 'uni_pc'):
            Solver used to sample the video.
        sampling_steps (`int`, *optional*, defaults to 50):
            Number of diffusion sampling steps. Higher values improve quality but slow generation
        denoise (`float`, *optional*, defaults to 1.0):
            Denoise strength. 1.0 means full denoising, 0.5 means half denoising
    """
    sampling_sigmas = None
    if sample_solver == "uni_pc":
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        if sigmas is None:
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift, denoise=denoise)
        else:
            sampling_sigmas = sigmas
            sample_scheduler.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * num_train_timesteps).to(torch.int64)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
    elif sample_solver == "dpm++":
        raise RuntimeError("Double shift operation may have issues; please check.")
        sample_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        if sigmas is None:
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift, denoise)
            retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * num_train_timesteps).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
    elif sample_solver == "euler":
        sample_scheduler = EulerScheduler(num_train_timesteps=num_train_timesteps, shift=shift, device=device)
        sample_scheduler.set_timesteps(sampling_steps, device=device, denoise=denoise, sigmas=sigmas)
    else:
        raise NotImplementedError(f"Unsupported solver type {sample_solver}.")
    return sample_scheduler, sampling_sigmas, sample_scheduler.timesteps


__all__ = ["get_sample_scheduler", SUPPORTED_SOLVERS]
