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

"""LatentHandler — latent 处理的默认实现。

Latent 相关方法，子类可覆写以定制行为。
"""

import torch

from kdit.utils import log


class LatentHandler:
    """Latent 的预处理、校验、pack/unpack 和辅助 latent 操作。"""

    def preprocess_base(self, base_latent_list: list[torch.Tensor]) -> torch.Tensor:
        """预处理 base_latent list，返回单个 Tensor。

        接收 [latent] 或 [latent, mask] 形式的 list，返回单个 Tensor。
        默认取第一个元素（latent）。
        子类可覆写此方法做模型特定的预处理，
        例如 Wan I2V 场景会将 [latent, mask] concat 为单个 tensor。
        """
        if len(base_latent_list) <= 0:
            raise ValueError("base_latent_list must contain at least one element")
        return base_latent_list[0]

    def validate_noise_shape(
        self,
        noise_shape: tuple[int] | list[int],
        diffusion_model,
        model_key,
    ) -> list[int]:
        """校验 noise_shape 为 4D（[vae_z_dim, f, h, w]）。"""
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

    def pack_noise(self, noise_latents: torch.Tensor, patch_size) -> torch.Tensor:
        """Pack noise latents，默认直通。"""
        return noise_latents

    def unpack_noise(self, noise_latents: torch.Tensor, patch_size) -> torch.Tensor:
        """Unpack noise latents，默认直通。"""
        return noise_latents

    def pack_aux(self, aux_latent: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Pack aux latent，默认直通。"""
        return aux_latent

    def apply_aux_latent(self, noise_latents, aux_latent, sample_config, timesteps, num_train_timesteps):
        """将 aux_latent 应用到 noise_latents，子类必须实现。"""
        raise NotImplementedError("subclass must implement apply_aux_latent method")

    def maybe_update_sample_config(self, sample_config, packed_noise_shape, default_settings):
        """按需更新 sample_config，默认直通。"""
        return sample_config
