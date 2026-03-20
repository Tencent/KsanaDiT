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

"""TextHandler — 文本 conditioning 处理的默认实现。

文本相关方法，子类可覆写以定制行为。
"""

import torch

from kdit.utils import log

from ..steps import tensor_ops


class TextHandler:
    """文本 conditioning 的预处理、校验、扩展和类型转换。"""

    def preprocess(self, conditioning: torch.Tensor | tuple):
        """预处理 text conditioning，默认直通返回。"""
        return conditioning

    def validate(self, positive: torch.Tensor, negative: torch.Tensor):
        """校验 positive / negative 为 3D tensor 且 batch 维度一致。"""
        log.info(
            f"positive shape:{positive.shape}, dtype:{positive.dtype}, device:{positive.device};"
            f" negtive shape:{negative.shape}, dtype:{negative.dtype}, device:{negative.device}"
        )
        if not (positive.ndim == negative.ndim == 3):
            raise ValueError(f"positive.shape {positive.shape}, negative.shape {negative.shape} must be 3D tensor")
        if positive.shape[0] != negative.shape[0]:
            raise ValueError(f"positive.shape[0] of {positive.shape}, negative.shape[0] of {negative.shape} must equal")
        return positive, negative

    def expand_to_batch(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        batch_size_per_prompts: list[int],
    ):
        """按 batch_size_per_prompts 扩展 positive / negative。"""
        if positive.shape[0] != negative.shape[0]:
            raise ValueError(f"positive.shape[0] of {positive.shape} must equal negative.shape[0] of {negative.shape}")
        positive = self._expand_to_total_prompts_size(positive, batch_size_per_prompts)
        negative = self._expand_to_total_prompts_size(negative, batch_size_per_prompts)
        return positive, negative

    def cast_to(self, positive, negative, *, dtype: torch.dtype, device: torch.device):
        """将 positive / negative 转换到目标 dtype 和 device。"""
        positive = tensor_ops.cast_to(positive, dtype=dtype, device=device)
        negative = tensor_ops.cast_to(negative, dtype=dtype, device=device)
        return positive, negative

    def get_num_prompts(self, text_tensor: torch.Tensor | tuple):
        """从 text_tensor 获取 prompt 数量（batch 维度大小）。"""
        if isinstance(text_tensor, tuple):
            text_tensor = text_tensor[0]
        if isinstance(text_tensor, torch.Tensor):
            return text_tensor.shape[0]
        else:
            raise ValueError("text_tensor must be torch.Tensor or tuple of torch.Tensor")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _expand_to_total_prompts_size(self, tensor: torch.Tensor, batch_size_per_prompts: list[int]):
        """按 batch_size_per_prompts repeat_interleave 扩展 tensor。"""
        num_prompts = tensor.shape[0]
        total_prompts_num = sum(batch_size_per_prompts)
        if num_prompts > total_prompts_num:
            raise ValueError(f"total_prompts_num({total_prompts_num}) must >= num_prompts({num_prompts})")
        if total_prompts_num > num_prompts:
            repeats = torch.tensor(batch_size_per_prompts, dtype=torch.int64, device=tensor.device)
            tensor = tensor.repeat_interleave(repeats, dim=0)
        return tensor
