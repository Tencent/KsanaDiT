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

"""QwenTextHandler — Qwen 模型的文本 conditioning 处理实现。"""

import torch
import torch.nn.functional as F

from kdit.utils import log

from ..handlers.text_handler import TextHandler


class QwenTextHandler(TextHandler):
    """Qwen 模型的文本 conditioning 处理，支持 (embeds, mask) tuple 格式。"""

    def preprocess(self, conditioning: torch.Tensor | tuple) -> tuple:
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

    def validate(self, positive: tuple, negative: tuple):
        pos, pos_mask = positive
        neg, neg_mask = negative
        log.info("text encoder tensor:")
        log.info(
            f"positive shape:{pos.shape}, dtype:{pos.dtype}, device:{pos.device};"
            f" negtive shape:{neg.shape}, dtype:{neg.dtype}, device:{neg.device}"
        )
        if not (pos.ndim == neg.ndim == 3):
            raise ValueError(f"positive.shape {pos.shape}, negative.shape {neg.shape} must be 3D tensor")
        if pos.shape[0] != neg.shape[0]:
            raise ValueError(f"positive.shape[0] of {pos.shape}, " f"negative.shape[0] of {neg.shape} must equal")

        log.info("text mask:")
        if not (pos_mask.ndim == neg_mask.ndim == 2):
            raise ValueError(f"pos_mask.shape {pos_mask.shape}, " f"neg_mask.shape {neg_mask.shape} must be 2D tensor")
        if pos_mask.shape[0] != neg_mask.shape[0]:
            raise ValueError(
                f"pos_mask.shape[0] of {pos_mask.shape}, " f"neg_mask.shape[0] of {neg_mask.shape} must equal"
            )
        return (pos, pos_mask), (neg, neg_mask)

    def expand_to_batch(
        self,
        positive: tuple,
        negative: tuple,
        batch_size_per_prompts: list[int],
    ):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos = self._expand_to_total_prompts_size(pos, batch_size_per_prompts)
        neg = self._expand_to_total_prompts_size(neg, batch_size_per_prompts)
        pos_mask = self._expand_to_total_prompts_size(pos_mask, batch_size_per_prompts)
        neg_mask = self._expand_to_total_prompts_size(neg_mask, batch_size_per_prompts)
        return (pos, pos_mask), (neg, neg_mask)

    def cast_to(
        self,
        positive: tuple,
        negative: tuple,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos = pos.to(dtype=dtype, device=device)
        neg = neg.to(dtype=dtype, device=device)
        # Mask tensors must also be on the same device for attention mask construction
        pos_mask = pos_mask.to(device=device)
        neg_mask = neg_mask.to(device=device)
        return (pos, pos_mask), (neg, neg_mask)

    def get_num_prompts(self, text_tensor: torch.Tensor | tuple):
        if isinstance(text_tensor, tuple):
            return text_tensor[0].shape[0]
        if isinstance(text_tensor, torch.Tensor):
            return text_tensor.shape[0]
        raise ValueError("text_tensor must be torch.Tensor or tuple of torch.Tensor")

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
