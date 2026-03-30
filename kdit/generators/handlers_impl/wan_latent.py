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

"""WanLatentHandler — Wan 模型的 latent 处理实现。"""

import torch

from kdit.config.sample_config import SampleConfig
from kdit.models.model_key import ModelKey
from kdit.utils import log

from ..handlers.latent_handler import LatentHandler


class WanLatentHandler(LatentHandler):
    """Wan 模型的 latent 预处理、校验和辅助 latent 操作。"""

    def __init__(self, model_key: ModelKey):
        self._model_key = model_key

    def preprocess_base(self, base_latent_list: list[torch.Tensor]) -> torch.Tensor:
        """Wan I2V: 将 [mask, latent] concat 为单个 tensor 作为模型的 y 输入。

        原先 base_latent 是已经 concat 好的单个 tensor（mask + latent 在 channel 维度），
        现在 BaseLatent 将 latent 和 mask 拆分存储，需要在此处重新 concat。
        T2V 场景下 list 只有 [latent]（无 mask），直接取第一个元素。
        """
        if self._model_key == ModelKey.Wan2_2_I2V_14B and len(base_latent_list) == 2:
            latent, mask = base_latent_list
            return torch.cat([mask, latent], dim=1)
        return base_latent_list[0]

    def validate_noise_shape(
        self,
        noise_shape: tuple[int] | list[int],
        diffusion_model,
        model_key: ModelKey,
    ) -> list[int]:
        """校验 noise_shape，I2V 模式下覆写 z_dim。"""
        log.info(f"input noise_shape: {noise_shape}")
        if not isinstance(noise_shape, (tuple, list)):
            raise ValueError(f"noise_shape {noise_shape} must be tuple or list")
        noise_shape = list(noise_shape)

        if len(noise_shape) != 4:
            raise ValueError(
                f"{model_key} noise_shape {noise_shape} dim must "
                "be 4 like:[vae_z_dim:16, f, h, w], f==1 when generate image"
            )
        return noise_shape

    def apply_aux_latent(
        self,
        noise_latents: torch.Tensor,
        aux_latent: torch.Tensor,
        sample_config: SampleConfig,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ):
        if aux_latent is None:
            return noise_latents

        if noise_latents.dim() != 5:  # [bs, z_dim, f, h, w]
            raise ValueError(f"noise_latents {noise_latents.shape} must be 5D tensor")

        aux_latent = aux_latent.to(noise_latents)
        frame_dim = 2
        if noise_latents.shape[frame_dim] < aux_latent.shape[frame_dim]:
            raise ValueError(
                f"noise_latents {noise_latents.shape} frame dim must be >= " f"aux_latent {aux_latent.shape}"
            )
        if aux_latent.shape[frame_dim] != noise_latents.shape[frame_dim]:
            aux_latent = torch.cat(
                [
                    aux_latent[:, :, :1].repeat(
                        1,
                        1,
                        noise_latents.shape[frame_dim] - aux_latent.shape[frame_dim],
                        1,
                        1,
                    ),
                    aux_latent,
                ],
                dim=frame_dim,
            )

        if sample_config.add_noise_to_latent:
            latent_timestep = timesteps[:1].to(noise_latents)
            noise_latents = (
                noise_latents * latent_timestep / num_train_timesteps
                + (1 - latent_timestep / num_train_timesteps) * aux_latent
            )
        else:
            noise_latents = aux_latent

        return noise_latents
