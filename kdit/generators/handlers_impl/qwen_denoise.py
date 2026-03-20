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

"""QwenDenoiseHandler — Qwen 模型的去噪循环钩子实现。"""

import torch

from kdit.models.model_key import ModelKey

from ..handlers.denoise_handler import DenoiseHandler
from .qwen_latent import QwenLatentHandler


class QwenDenoiseHandler(DenoiseHandler):
    """Qwen 模型的去噪循环钩子，支持 text padding 和 Edit 模式 norm rescale。"""

    def __init__(self, model_key: ModelKey, latent_handler: QwenLatentHandler):
        self._model_key = model_key
        self._latent_handler = latent_handler

    def prepare_model_forward_kargs(
        self,
        cfg_scale: float,
        *,
        positive,
        negative,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        base_latent,
        aux_latent=None,
        **_,
    ) -> dict | tuple[dict, dict]:
        if cache is not None:
            raise NotImplementedError(f"{self._model_key} does not support cache yet!")
        base = {"cache": cache, "step_iter": step_iter}
        positive_embeds, positive_mask = positive
        negative_embeds, negative_mask = negative

        img_shapes = self._latent_handler._get_latent_img_shapes()
        positive_txt_seq_lens = positive_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_mask.sum(dim=1).tolist()
        positive_embeds, positive_mask, negative_embeds, negative_mask = self._pad_text_pair(
            positive_embeds, positive_mask, negative_embeds, negative_mask
        )
        use_cfg = abs(cfg_scale - 1.0) > 1e-6

        # aux_latent 是 VAE encode 的参考图 latent（Edit 模式），
        # base_latent 是空 latent（T2I 模式）
        if use_cfg and combine_cond_uncond:
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_embs = torch.cat([positive_embeds, negative_embeds], dim=0)
            combine_mask = torch.cat([positive_mask, negative_mask], dim=0)
            combine_txt_seq_lens = positive_txt_seq_lens + negative_txt_seq_lens
            combine_img_shapes = [list(shapes) for shapes in img_shapes] + [list(shapes) for shapes in img_shapes]

            combine_aux = aux_latent
            if aux_latent is not None:
                combine_aux = [torch.cat([r, r], dim=0) for r in aux_latent]

            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "img_shapes": combine_img_shapes,
                "encoder_hidden_states": combine_embs,
                "encoder_hidden_states_mask": combine_mask,
                "txt_seq_lens": combine_txt_seq_lens,
                "aux_latents": combine_aux,
            }
            return base | combine_kargs

        base.update(
            {
                "x": noise_latent,
                "t": timestep,
                "img_shapes": img_shapes,
                "aux_latents": aux_latent,
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

    def apply_cfg(self, cfg_scale, cond, uncond, **kwargs):
        comb_pred = uncond + float(cfg_scale) * (cond - uncond)
        if self._model_key == ModelKey.QwenImage_Edit:
            # Normalize to conditional prediction norm (per-token, last-dim),
            # matching diffusers.
            cond_norm = torch.norm(cond, dim=-1, keepdim=True)
            comb_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            scale = (cond_norm / comb_norm).to(dtype=comb_pred.dtype)
            comb_pred = comb_pred * scale
        return comb_pred

    @staticmethod
    def _pad_text_pair(embeds_a, mask_a, embeds_b, mask_b):
        """Pad text embeddings and masks to the same sequence length."""
        import torch.nn.functional as F

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
