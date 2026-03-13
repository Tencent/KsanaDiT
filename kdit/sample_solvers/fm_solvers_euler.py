# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from lightning and diffuers

import numpy as np
import torch

# pyright: ignore
from diffusers import FlowMatchEulerDiscreteScheduler  # pyright: ignore
from torch import Tensor

from ..utils.sample_solver import apply_sigma_shift, get_sigmas_with_denoise


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


class FlowMatchEulerScheduler:
    """
    adopted from diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
    and diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        device: torch.device | str = "cuda",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.device = device
        self.init_noise_sigma = 1.0

        self.timesteps: Tensor = None
        self.sigmas: Tensor = None
        self._step_index: int = None

    @staticmethod
    def _time_shift_exponential(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        return np.exp(mu) / (np.exp(mu) + np.power((1.0 / t - 1.0), sigma))

    def index_for_timestep(self, timestep: Tensor, schedule_timesteps: Tensor | None = None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | None = None,
        shift: float = None,
        denoise: float = 1.0,
    ):
        device = device or self.device
        mu = shift if shift is not None else self.shift

        steps = int(max(num_inference_steps, 1))
        base_end = 1.0 / float(steps)
        base_sigmas = np.linspace(1.0, base_end, steps, dtype=np.float32)

        if denoise is not None and denoise < 0.9999 and denoise > 0.0:
            new_steps = int(steps / denoise)
            full = np.linspace(1.0, 1.0 / float(new_steps), new_steps, dtype=np.float32)
            base_sigmas = full[-steps:]

        shifted = self._time_shift_exponential(mu=float(mu), sigma=1.0, t=base_sigmas)
        shifted = np.concatenate([shifted, np.array([0.0], dtype=np.float32)])

        self.sigmas = torch.from_numpy(shifted).to(dtype=torch.float32, device=device)
        self.timesteps = self.sigmas[:-1] * float(self.num_train_timesteps)
        self._step_index = None

    def _init_step_index(self, timestep: Tensor):
        try:
            self._step_index = self.index_for_timestep(timestep)
        except Exception:  # pylint: disable=broad-except
            self._step_index = (self.timesteps - timestep).abs().argmin().item()

    @property
    def step_index(self) -> int:
        return self._step_index

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor,
        sample: Tensor,
        **kwargs,
    ) -> Tensor:
        if self._step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas[self._step_index].to(sample.device)
        sigma_next = self.sigmas[self._step_index + 1].to(sample.device)

        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
            sigma_next = sigma_next.unsqueeze(-1)

        prev_sample = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return prev_sample

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        sigmas = timesteps / self.num_train_timesteps
        while sigmas.ndim < original_samples.ndim:
            sigmas = sigmas.unsqueeze(-1)
        return (1 - sigmas) * original_samples + sigmas * noise

    def get_velocity(
        self,
        sample: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        return noise - sample

    def scale_noise(
        self,
        sample: Tensor,
        timestep: Tensor,
        noise: Tensor,
    ) -> Tensor:
        sigma = timestep / self.num_train_timesteps
        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * sample + sigma * noise
