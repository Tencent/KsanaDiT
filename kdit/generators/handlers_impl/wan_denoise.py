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

"""WanDenoiseHandler — Wan 模型的去噪循环钩子实现。"""

import torch

from kdit.models.diffusion_model import KsanaDiffusionModel
from kdit.models.model_key import ModelKey
from kdit.utils import log

from ..handlers.denoise_handler import DenoiseHandler


class WanDenoiseHandler(DenoiseHandler):
    """Wan 模型的去噪循环钩子，支持双模型 boundary 切换。"""

    def __init__(self):
        self._boundary = None

    def _get_model_boundary(self, diffusion_model: list[KsanaDiffusionModel]):
        if self._boundary is not None:
            return self._boundary
        if len(diffusion_model) < 2:
            return None

        default_settings = diffusion_model[0].default_settings
        high_model, low_model = diffusion_model
        self._boundary = None
        if low_model is not None:
            input_boundary = getattr(high_model.model_config, "boundary", None)
            default_boundary = getattr(default_settings.runtime_config, "boundary", None)
            boundary = input_boundary or default_boundary
            if boundary is None:
                raise RuntimeError("boundary should be set when low_model is not None")
            num_train_timesteps = getattr(default_settings.sample_config, "num_train_timesteps", None)
            if num_train_timesteps is None:
                raise RuntimeError("num_train_timesteps should be set in yaml sample_config settings")
            self._boundary = boundary * num_train_timesteps
            log.info(f"model boundary: {boundary}")
        return self._boundary

    def get_running_model(
        self,
        diffusion_model: list[KsanaDiffusionModel],
        *,
        timestep_id: int,
        device: torch.device,
        offload_device: torch.device,
    ):
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

    def get_running_cache(self, dit_cache, *, timestep_id: int):
        if not isinstance(dit_cache, (list, tuple)):
            return dit_cache
        if len(dit_cache) == 1:
            return dit_cache[0]
        if len(dit_cache) != 2:
            raise ValueError(f"dit_cache must be list of 1 or 2 float, but got {dit_cache}")

        high_cache, low_cache = dit_cache
        if low_cache is None:
            return high_cache
        if timestep_id >= self._boundary:
            return high_cache
        else:
            high_cache.offload_to_cpu()
            return low_cache

    def get_running_cfg_scale(self, cfg_scale: list[float], *, timestep_id: int):
        if not isinstance(cfg_scale, (list, tuple)):
            return cfg_scale
        if len(cfg_scale) == 1:
            return cfg_scale[0]
        if len(cfg_scale) != 2:
            raise ValueError(f"cfg_scales must be list of 1 or 2 float, but got {cfg_scale}")
        if cfg_scale[1] is not None and self._boundary is not None and timestep_id < self._boundary:
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
        base_latent: torch.Tensor | None,
        model_key: ModelKey,
        aux_latent=None,
        **_,
    ) -> dict | tuple[dict, dict]:
        base = {"cache": cache, "step_iter": step_iter}

        use_cfg = abs(cfg_scale - 1.0) > 1e-6
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
            if model_key == ModelKey.Wan2_2_I2V_14B and base_latent is not None:
                combine_kargs["y"] = torch.cat([base_latent, base_latent], dim=0)
            return base | combine_kargs

        base.update({"x": noise_latent, "t": timestep})
        arg_cond = {"phase": "cond", "context": positive}
        arg_uncond = {"phase": "uncond", "context": negative}
        if model_key == ModelKey.Wan2_2_I2V_14B:
            arg_cond["y"] = base_latent
            arg_uncond["y"] = base_latent
        if use_cfg:
            return base | arg_cond, base | arg_uncond
        else:
            return base | arg_cond
