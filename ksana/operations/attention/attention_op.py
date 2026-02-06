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
import torch.nn as nn

from ksana.config import KsanaAttentionConfig

from ...utils.logger import log
from .backends import KsanaAttentionBackendImpl, get_attention_backend_impl


class KsanaAttentionOp(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        *,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        attention_config: KsanaAttentionConfig = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5
        self.causal = causal
        if attention_config is None:
            raise ValueError("attention_config should not be None")
        self.attention_config = attention_config
        log.debug(f"KsanaAttentionOp with config: {self.attention_config}")

        self._attn_impl: KsanaAttentionBackendImpl | None = None
        self._attn_impl_dtype: torch.dtype | None = None

    @property
    def backend_type(self) -> str | None:
        return self._attn_impl.type() if self._attn_impl else None

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        if self._attn_impl is not None and dtype == self._attn_impl_dtype:
            return

        backend_impl = get_attention_backend_impl(
            attention_config=self.attention_config,
            head_size=self.head_size,
            dtype=dtype,
        )
        self._attn_impl = backend_impl(
            attention_config=self.attention_config,
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
        )
        self._attn_impl_dtype = dtype

    @torch.compiler.disable
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **backend_kwargs,
    ) -> torch.Tensor:
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("LocalAttentionOp expects tensors shaped [B, L, H, D].")

        self._ensure_impl(query.dtype)
        assert self._attn_impl is not None

        return self._attn_impl.forward(
            query,
            key,
            value,
            **backend_kwargs,
        )
