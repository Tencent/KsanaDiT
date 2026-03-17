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
import torch.nn.functional as F

from kdit.config import SampleConfig
from kdit.config.wan_experimental_config import KsanaExperimentalConfig
from kdit.models import ModelKey
from kdit.utils import evolve_with_recommend, log

from .base_generator import BaseGenerator
from .generator_factory import GeneratorFactory


@GeneratorFactory.register([ModelKey.QwenImage_T2I, ModelKey.QwenImage_Edit])
class QwenGenerator(BaseGenerator):

    def __init__(self):
        super().__init__()
        self.latent_img_shapes = None

    def preprocess_text_conditioning(self, conditioning: torch.Tensor | tuple) -> tuple:
        if isinstance(conditioning, (tuple, list)) and len(conditioning) == 2:
            embeds, mask = conditioning
            if isinstance(mask, torch.Tensor):
                return embeds, mask
        # ComfyUI format: only embeddings tensor provided
        if isinstance(conditioning, torch.Tensor):
            embeds = conditioning
            # Generate all-ones attention mask based on sequence length
            batch_size, seq_len = embeds.shape[:2]
            mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=embeds.device)
            return embeds, mask
        raise ValueError(f"Unsupported conditioning format: {type(conditioning)}")

    def preprocess_image_embeds(self, image_embeds):
        """上游已统一为 list[Tensor]，此处仅做透传。"""
        return image_embeds

    def _valid_prompts(self, positive: tuple, negative: tuple):
        pos, pos_mask = positive
        neg, neg_mask = negative
        log.info("text encoder tensor:")
        pos, neg = super()._valid_prompts(pos, neg)

        log.info("text mask:")
        if not (pos_mask.ndim == neg_mask.ndim == 2):
            raise ValueError(f"pos_mask.shape {positive.shape}, neg_mask.shape {negative.shape} must be 2D tensor")
        if pos_mask.shape[0] != neg_mask.shape[0]:
            raise ValueError(f"pos_mask.shape[0] of {positive.shape}, neg_mask.shape[0] of {negative.shape} must equal")
        return (pos, pos_mask), (neg, neg_mask)

    def _valid_prompts_to_total_prompts_size(
        self,
        positive: tuple,
        negative: tuple,
        batch_size_per_prompts: list[int],
    ):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos, neg = super()._valid_prompts_to_total_prompts_size(pos, neg, batch_size_per_prompts)
        pos_mask, neg_mask = super()._valid_prompts_to_total_prompts_size(pos_mask, neg_mask, batch_size_per_prompts)
        neg = self._expand_to_total_prompts_size(neg, batch_size_per_prompts)
        pos_mask = self._expand_to_total_prompts_size(pos_mask, batch_size_per_prompts)
        neg_mask = self._expand_to_total_prompts_size(neg_mask, batch_size_per_prompts)
        return (pos, pos_mask), (neg, neg_mask)

    def cast_text_tensors_to(self, positive: tuple, negative: tuple, *, dtype: torch.dtype, device: torch.device):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos, neg = super().cast_text_tensors_to(pos, neg, dtype=dtype, device=device)
        # Mask tensors must also be on the same device for attention mask construction
        pos_mask = pos_mask.to(device=device)
        neg_mask = neg_mask.to(device=device)
        return (pos, pos_mask), (neg, neg_mask)

    def apply_cfg(
        self,
        cfg_scale,
        cond,
        uncond,
        experimental_config: KsanaExperimentalConfig | None = None,
        step_index: int = 0,
        total_steps: int = 1,
        **kwargs,
    ):
        if experimental_config is not None:
            return super().apply_cfg(
                cfg_scale,
                cond,
                uncond,
                experimental_config=experimental_config,
                step_index=step_index,
                total_steps=total_steps,
            )
        comb_pred = uncond + float(cfg_scale) * (cond - uncond)
        if self.model_key == ModelKey.QwenImage_Edit:
            # Normalize to conditional prediction norm (per-token, last-dim), matching diffusers.
            cond_norm = torch.norm(cond, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            scale = (cond_norm / comb_norm).to(dtype=comb_pred.dtype)
            comb_pred = comb_pred * scale
        return comb_pred

    def calculate_shift(self, seq_len: int, configs) -> float:
        base_seq_len = getattr(configs, "base_seq_len", 256)
        max_seq_len = getattr(configs, "max_seq_len", 4096)
        base_shift = getattr(configs, "base_shift", 0.5)
        max_shift = getattr(configs, "max_shift", 1.15)

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return seq_len * m + b

    def prepare_model_forward_kargs(
        self,
        cfg_scale,
        *,
        positive,
        negative,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        image_embeds,
        **_,
    ) -> dict | tuple[dict, dict]:
        if cache is not None:
            raise NotImplementedError(f"{self.model_key} does not support cache yet!")
        base = {"cache": cache, "step_iter": step_iter}
        positive_embeds, positive_mask = positive
        negative_embeds, negative_mask = negative

        img_shapes = self._get_latent_img_shapes()
        positive_txt_seq_lens = positive_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_mask.sum(dim=1).tolist()
        positive_embeds, positive_mask, negative_embeds, negative_mask = self._pad_text_pair(
            positive_embeds, positive_mask, negative_embeds, negative_mask
        )
        use_cfg = self._use_cfg(cfg_scale)

        ref_latents = image_embeds
        if use_cfg and combine_cond_uncond:
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_embs = torch.cat([positive_embeds, negative_embeds], dim=0)
            combine_mask = torch.cat([positive_mask, negative_mask], dim=0)
            combine_txt_seq_lens = positive_txt_seq_lens + negative_txt_seq_lens
            combine_img_shapes = [list(shapes) for shapes in img_shapes] + [list(shapes) for shapes in img_shapes]

            combine_ref_latents = ref_latents
            if ref_latents is not None:
                combine_ref_latents = [torch.cat([r, r], dim=0) for r in ref_latents]

            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "img_shapes": combine_img_shapes,
                "encoder_hidden_states": combine_embs,
                "encoder_hidden_states_mask": combine_mask,
                "txt_seq_lens": combine_txt_seq_lens,
                "ref_latents": combine_ref_latents,
            }
            return base | combine_kargs

        base.update(
            {
                "x": noise_latent,
                "t": timestep,
                "img_shapes": img_shapes,
                "ref_latents": ref_latents,
            }
        )
        arg_cond = {
            "phase": "cond",
            "encoder_hidden_states": positive_embeds,
            "encoder_hidden_states_mask": positive_mask,
            "txt_seq_lens": positive_txt_seq_lens,
        }
        if not use_cfg:
            return base | arg_cond
        arg_uncond = {
            "phase": "uncond",
            "encoder_hidden_states": negative_embeds,
            "encoder_hidden_states_mask": negative_mask,
            "txt_seq_lens": negative_txt_seq_lens,
        }
        return base | arg_cond, base | arg_uncond

    def _apply_input_latent(
        self,
        noise_latents: torch.Tensor,
        input_latent: torch.Tensor,
        sample_config: SampleConfig,
        timesteps: torch.Tensor,
        num_train_timesteps: int,
    ):
        # TODO: implement input_latent blending for image editing
        log.warning(
            "input_latent blending for image editing is not implemented yet. "
            "Currently input_latent is not used for qwen, mainly for getting output shape"
        )
        return noise_latents

    def _get_latent_img_shapes(self):
        if self.latent_img_shapes is None:
            raise ValueError("latent_img_shapes is None, please call pack_noise_latents first to set it")
        return self.latent_img_shapes

    def pack_noise_latents(self, latents, patch_size):
        if latents.dim() != 5:
            raise ValueError(f"{self.model_key} pack latents {latents.shape} must be 5D tensor")
        num, z_dim, latent_f, latent_h, latent_w = latents.shape
        if latent_f != 1:
            raise ValueError(f"{self.model_key} pack latents latent_f  must be 1, but got {latent_f}")
        latents = latents.squeeze(2)  # remove latent_f dim
        latents = latents.view(num, z_dim, latent_h // patch_size, patch_size, latent_w // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        new_h = latent_h // patch_size
        new_w = latent_w // patch_size
        latents = latents.reshape(num, new_h * new_w, z_dim * patch_size * patch_size)

        self.latent_img_shapes = [[(1, new_h, new_w)] for _ in range(num)]
        log.info(f"{self.model_key} pack noise latents to shape {latents.shape}")
        return latents

    def pack_ref_latents(self, ref_latents: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]:
        packed_refs = []
        for ref in ref_latents:
            latent = ref.squeeze(2) if ref.dim() == 5 else ref
            if latent.dim() == 4:
                b, c, h, w = latent.shape
                latent = latent.view(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
                latent = latent.permute(0, 2, 4, 1, 3, 5).reshape(
                    b, (h // patch_size) * (w // patch_size), c * patch_size * patch_size
                )
            packed_refs.append(latent)
            h, w = ref.shape[-2], ref.shape[-1]
            ref_shape = (1, h // patch_size, w // patch_size)
            for i in range(len(self.latent_img_shapes)):
                self.latent_img_shapes[i].append(ref_shape)

        log.info(f"{self.model_key} packed {len(ref_latents)} reference latents")
        return packed_refs

    def unpack_noise_latents(self, latents, patch_size):
        if latents.dim() != 3:
            raise ValueError(f"{self.model_key} unpack latents input {latents.shape} must be 3D tensor")
        num, hw, z_dim = latents.shape
        img_shapes = self._get_latent_img_shapes()
        _, new_h, new_w = img_shapes[0][0]
        latents = latents.view(num, new_h, new_w, z_dim // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(num, z_dim // (patch_size * patch_size), 1, new_h * patch_size, new_w * patch_size)
        log.info(f"{self.model_key} unpack noise latents shape from {(num, hw, z_dim)} to {latents.shape}")
        return latents

    def maybe_update_sample_config(self, sample_config: SampleConfig, packed_noise_shape: list, default_settings):
        if sample_config.shift is not None:
            return sample_config

        if len(packed_noise_shape) != 3:
            raise RuntimeError(f"packed_noise_shape {packed_noise_shape} should be 3D")
        mu = self.calculate_shift(packed_noise_shape[1], default_settings.sample_config)
        sample_config = evolve_with_recommend(sample_config, {"shift": mu})
        log.info(f"update sample_config shift to {sample_config}")
        return sample_config

    def _pad_text_pair(self, embeds_a, mask_a, embeds_b, mask_b):
        max_txt_len = max(embeds_a.shape[1], embeds_b.shape[1])
        pad_a = max_txt_len - embeds_a.shape[1]
        pad_b = max_txt_len - embeds_b.shape[1]
        if pad_a > 0:
            embeds_a = F.pad(embeds_a, (0, 0, 0, pad_a))
            mask_a = F.pad(mask_a, (0, pad_a))
        if pad_b > 0:
            embeds_b = F.pad(embeds_b, (0, 0, 0, pad_b))
            mask_b = F.pad(mask_b, (0, pad_b))
        return embeds_a, mask_a, embeds_b, mask_b
