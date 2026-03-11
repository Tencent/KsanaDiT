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

import torch

from ksana.config import KsanaSampleConfig
from ksana.models import KsanaDiffusionModel, KsanaModelKey
from ksana.utils import log

from ..base_unit import KsanaUnitFactory, KsanaUnitType
from .base_generator import KsanaBaseGenerator


# TODO: need better abstract base implement for vace, vace can not invade base
@KsanaUnitFactory.register(
    KsanaUnitType.GENERATOR,
    [KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.Wan2_2_I2V_14B],
)
class KsanaWanGenerator(KsanaBaseGenerator):
    def __init__(self):
        super().__init__()
        # TODO: maybe could remove boundary, use allow each model input steps instead
        self.boundary = None

    def valid_noise_shape(self, noise_shape: tuple[int] | list[int], diffusion_model: list[KsanaDiffusionModel]):
        noise_shape = super().valid_noise_shape(noise_shape, diffusion_model)
        if self.model_key == KsanaModelKey.Wan2_2_I2V_14B:
            # Note: i2v used image_embeds as noise_shape, so need change to shape[1] as right z_dim
            #       and should have added z_dim to yaml settings
            default_settings = diffusion_model[0].default_settings
            if not hasattr(default_settings.vae, "z_dim"):
                raise ValueError("vae.z_dim not found in default_model_settings.vae")
            noise_shape[0] = default_settings.vae.z_dim
        return noise_shape

    def cast_image_tensor_to(
        self, image_embeds: list[torch.Tensor] | None, *, dtype: torch.dtype, device: torch.device
    ):
        if self.model_key == KsanaModelKey.Wan2_2_T2V_14B:
            return None
        return super().cast_image_tensor_to(image_embeds, dtype=dtype, device=device)

    def _get_model_boundary(self, diffusion_model: list[KsanaDiffusionModel]):
        if self.boundary is not None:
            return self.boundary
        if len(diffusion_model) < 2:
            return None

        default_settings = diffusion_model[0].default_settings
        high_model, low_model = diffusion_model
        self.boundary = None
        if low_model is not None:
            input_boundary = getattr(high_model.model_config, "boundary", None)
            default_boundary = getattr(default_settings.runtime_config, "boundary", None)
            boundary = input_boundary or default_boundary
            if boundary is None:
                raise RuntimeError("boundary should be set when low_model is not None")
            self.boundary = boundary * self._get_num_train_timesteps(default_settings)
            log.info(f"model boundary: {boundary}")
        return self.boundary

    def _apply_input_latent(
        self,
        noise_latents: torch.Tensor,
        input_latent: torch.Tensor,
        sample_config: KsanaSampleConfig,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ):
        if input_latent is None:
            return noise_latents

        if noise_latents.dim() != 5:  # [bs, z_dim, f, h, w]
            raise ValueError(f"noise_latents {noise_latents.shape} must be 5D tensor")

        input_latent = input_latent.to(noise_latents)
        frame_dim = 2
        if noise_latents.shape[frame_dim] < input_latent.shape[frame_dim]:
            raise ValueError(
                f"noise_latents {noise_latents.shape} frame dim must be >= input_latent {input_latent.shape}"
            )
        if input_latent.shape[frame_dim] != noise_latents.shape[frame_dim]:
            input_latent = torch.cat(
                [
                    input_latent[:, :, :1].repeat(
                        1, 1, noise_latents.shape[frame_dim] - input_latent.shape[frame_dim], 1, 1
                    ),
                    input_latent,
                ],
                dim=frame_dim,
            )

        if sample_config.add_noise_to_latent:
            latent_timestep = timesteps[:1].to(noise_latents)
            noise_latents = (
                noise_latents * latent_timestep / num_train_timesteps
                + (1 - latent_timestep / num_train_timesteps) * input_latent
            )
        else:
            noise_latents = input_latent

        return noise_latents

    def get_running_model(self, diffusion_model, timestep_id: int, device=None, offload_device=None):
        if device is None:
            raise ValueError("device must be provided")
        if not isinstance(diffusion_model, (list, tuple)):
            raise RuntimeError(f"diffusion_model must be a list but got {diffusion_model}")
        if len(diffusion_model) == 1:
            return diffusion_model[0]
        if len(diffusion_model) != 2:
            raise ValueError(f"diffusion_model must be list of 1 or 2 float, but got {diffusion_model}")
        high_model, low_model = diffusion_model
        boundary = self._get_model_boundary(diffusion_model)
        if low_model is not None and boundary is None:
            raise ValueError("boundary must be provided when low_model is not None")
        use_high = low_model is None or (boundary is not None and timestep_id >= boundary)
        if use_high:
            if low_model is not None:
                if low_model.device != offload_device:
                    low_model.to(offload_device)
            return high_model
        else:
            if high_model.device != offload_device:
                high_model.to(offload_device)
            return low_model

    def get_running_cache(self, dit_cache, timestep_id):
        if not isinstance(dit_cache, (list, tuple)):
            return dit_cache
        if len(dit_cache) == 1:
            return dit_cache[0]
        if len(dit_cache) != 2:
            raise ValueError(f"dit_cache must be list of 1 or 2 float, but got {dit_cache}")

        high_cache, low_cache = dit_cache
        if low_cache is None:
            return high_cache
        if timestep_id >= self.boundary:
            return high_cache
        else:
            high_cache.offload_to_cpu()
            return low_cache

    def get_running_cfg_scale(self, cfg_scale: list[float], timestep_id: int):
        if not isinstance(cfg_scale, (list, tuple)):
            return cfg_scale
        if len(cfg_scale) == 1:
            return cfg_scale[0]
        if len(cfg_scale) != 2:
            raise ValueError(f"cfg_scales must be list of 1 or 2 float, but got {cfg_scale}")
        if cfg_scale[1] is not None and self.boundary is not None and timestep_id < self.boundary:
            return cfg_scale[1]
        else:
            return cfg_scale[0]

    def prepare_model_forward_kargs(
        self,
        cfg_scale: float,
        *,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        positive,
        negative,
        image_embeds: list[torch.Tensor] | None,
        **_,
    ) -> dict:
        base = {"cache": cache, "step_iter": step_iter}

        # Wan I2V: image_embeds is list[Tensor] with single element; extract the Tensor for model "y" input
        img_y = image_embeds[0] if image_embeds is not None and len(image_embeds) > 0 else None

        use_cfg = self._use_cfg(cfg_scale)
        if use_cfg and combine_cond_uncond:
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_context = torch.cat([positive, negative], dim=0)
            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "context": combine_context,
            }
            if self.model_key == KsanaModelKey.Wan2_2_I2V_14B and img_y is not None:
                combine_kargs["y"] = torch.cat([img_y, img_y], dim=0)
            return base | combine_kargs

        base.update({"x": noise_latent, "t": timestep})
        arg_cond = {"phase": "cond", "context": positive}
        arg_uncond = {"phase": "uncond", "context": negative}
        if self.model_key == KsanaModelKey.Wan2_2_I2V_14B:
            arg_cond["y"] = img_y
            arg_uncond["y"] = img_y
        if use_cfg:
            return base | arg_cond, base | arg_uncond
        else:
            return base | arg_cond
