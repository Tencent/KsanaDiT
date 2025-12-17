# from lightning and diffuers

import torch

# pyright: ignore
from diffusers import FlowMatchEulerDiscreteScheduler  # pyright: ignore
from torch import Tensor
import numpy as np
from ..utils.sample_solver import get_sigmas_with_denoise, apply_sigma_shift


def unsqueeze_to_ndim(in_tensor: Tensor, tgt_n_dim: int):
    if in_tensor.ndim > tgt_n_dim:

        return in_tensor
    if in_tensor.ndim < tgt_n_dim:
        in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
    return in_tensor


class EulerScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self, num_train_timesteps: int, shift: float = 1.0, device: torch.device | str = "cuda", **kwargs
    ) -> None:
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift, **kwargs)
        self.init_noise_sigma = 1.0
        self.num_train_timesteps = num_train_timesteps
        self._shift = shift
        self.init_noise_sigma = 1.0
        self.device = device
        self.set_timesteps(num_inference_steps=num_train_timesteps)
        pass

    def set_shift(self, shift: float = 1.0):
        self.sigmas = apply_sigma_shift(self.sigmas, shift)
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        self._shift = shift

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | int | None = None,
        denoise: float = 1.0,
        sigmas: Tensor | list[float] | None = None,
    ):
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device or self.device)
        else:
            sigmas_array = get_sigmas_with_denoise(
                steps=num_inference_steps,
                denoise=denoise,
                start=1,
                end=0,
            )
            sigmas_with_zero = np.concatenate(
                [sigmas_array, [0.0]]
            )  # 添加最后的 0.0，使其与传入的 sigmas 数量保持一致，都是steps+1个value
            self.sigmas = torch.from_numpy(sigmas_with_zero).to(dtype=torch.float32, device=device or self.device)
        self.set_shift(self._shift)
        self._step_index = None
        self._begin_index = None

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        **kwargs,
    ) -> tuple:
        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)  # pyright: ignore
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device)
        x_t_next = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return x_t_next
