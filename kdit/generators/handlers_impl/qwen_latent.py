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

"""QwenLatentHandler — Qwen 模型的 latent 处理实现。"""

import torch

from kdit.config.sample_config import SampleConfig
from kdit.utils import evolve_with_recommend, log

from ..handlers.latent_handler import LatentHandler


class QwenLatentHandler(LatentHandler):
    """Qwen 模型的 latent pack/unpack、辅助 latent 和 sample_config 更新。"""

    def __init__(self):
        self._latent_img_shapes: list = []

    def pack_noise(self, latents: torch.Tensor, patch_size: int) -> torch.Tensor:
        if latents.dim() != 5:
            raise ValueError(f"pack latents {latents.shape} must be 5D tensor")
        num, z_dim, latent_f, latent_h, latent_w = latents.shape
        if latent_f != 1:
            raise ValueError(f"pack latents latent_f must be 1, but got {latent_f}")
        latents = latents.squeeze(2)  # remove latent_f dim
        latents = latents.view(num, z_dim, latent_h // patch_size, patch_size, latent_w // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        new_h = latent_h // patch_size
        new_w = latent_w // patch_size
        latents = latents.reshape(num, new_h * new_w, z_dim * patch_size * patch_size)

        self._latent_img_shapes = [[(1, new_h, new_w)] for _ in range(num)]
        log.info(f"pack noise latents to shape {latents.shape}")
        return latents

    def unpack_noise(self, latents: torch.Tensor, patch_size: int) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"unpack latents input {latents.shape} must be 3D tensor")
        num, hw, z_dim = latents.shape
        img_shapes = self._get_latent_img_shapes()
        _, new_h, new_w = img_shapes[0][0]
        latents = latents.view(num, new_h, new_w, z_dim // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(num, z_dim // (patch_size * patch_size), 1, new_h * patch_size, new_w * patch_size)
        log.info(f"unpack noise latents shape from {(num, hw, z_dim)} to {latents.shape}")
        return latents

    def pack_aux(self, aux_latent, patch_size: int):
        """Pack aux latent（参考图 latent 列表）。"""
        if aux_latent is None:
            return aux_latent
        return self._pack_aux_latents(aux_latent, patch_size)

    def apply_aux_latent(
        self,
        noise_latents: torch.Tensor,
        aux_latent,
        sample_config: SampleConfig,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ):
        # Qwen 不在噪声混合阶段使用 aux_latent；aux_latent
        # 通过 prepare_model_forward_kargs 的 aux_latent 参数直接传递给模型。
        return noise_latents

    def _pack_aux_latents(self, aux_latents: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]:
        packed_refs = []
        for ref in aux_latents:
            latent = ref.squeeze(2) if ref.dim() == 5 else ref
            if latent.dim() == 4:
                b, c, h, w = latent.shape
                latent = latent.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                latent = latent.permute(0, 2, 4, 1, 3, 5).reshape(
                    b,
                    (h // patch_size) * (w // patch_size),
                    c * patch_size * patch_size,
                )
            packed_refs.append(latent)
            h, w = ref.shape[-2], ref.shape[-1]
            ref_shape = (1, h // patch_size, w // patch_size)
            for i in range(len(self._latent_img_shapes)):
                self._latent_img_shapes[i].append(ref_shape)

        log.info(f"packed {len(aux_latents)} aux latents")
        return packed_refs

    def maybe_update_sample_config(self, sample_config: SampleConfig, packed_noise_shape: list, default_settings):
        if sample_config.shift is not None:
            return sample_config

        if len(packed_noise_shape) != 3:
            raise RuntimeError(f"packed_noise_shape {packed_noise_shape} should be 3D")
        mu = self.calculate_shift(packed_noise_shape[1], default_settings.sample_config)
        sample_config = evolve_with_recommend(sample_config, {"shift": mu})
        log.info(f"update sample_config shift to {sample_config}")
        return sample_config

    def calculate_shift(self, seq_len: int, configs) -> float:
        base_seq_len = getattr(configs, "base_seq_len", 256)
        max_seq_len = getattr(configs, "max_seq_len", 4096)
        base_shift = getattr(configs, "base_shift", 0.5)
        max_shift = getattr(configs, "max_shift", 1.15)

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return seq_len * m + b

    def _get_latent_img_shapes(self):
        if not self._latent_img_shapes:
            raise ValueError("latent_img_shapes is empty, please call pack_noise first to set it")
        return self._latent_img_shapes
