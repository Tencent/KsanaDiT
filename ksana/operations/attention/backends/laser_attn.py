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

from ksana.accelerator import platform
from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig

from .base import KsanaAttentionBackendImpl

_MIND_IE_SD_AVAILABLE = False

if platform.is_npu():
    import torch_npu  # pylint: disable=unused-import # noqa: F401

    try:
        from mindiesd import attention_forward

        _MIND_IE_SD_AVAILABLE = True
    except ImportError:
        attention_forward = None


class LaserAttentionImpl(KsanaAttentionBackendImpl):
    @staticmethod
    def type() -> KsanaAttentionBackend:
        return KsanaAttentionBackend.LASER_ATTN

    @staticmethod
    def supports(**_) -> bool:
        return _MIND_IE_SD_AVAILABLE

    def __init__(
        self,
        attention_config: KsanaAttentionConfig,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_config = attention_config
        self.check_config()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        q_lens: torch.Tensor = None,
        k_lens: torch.Tensor = None,
        dropout_p: float = None,
        softmax_scale: float = None,
        **_,
    ) -> torch.Tensor:
        if not _MIND_IE_SD_AVAILABLE:
            raise RuntimeError("LaserAttention is not supported, try to install mindiesd")
        if dropout_p is not None:
            raise RuntimeError("dropout is not supported by LaserAttention")
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        seq_dim = 1  # q, k, v layout: BSND
        if query.shape[seq_dim] == key.shape[seq_dim]:
            compute_layout = "BNSD"  # only the compute layout, not the input layout, input layout is always BSND
            out = attention_forward(
                query,
                key,
                value,
                scale=scale,
                opt_mode="manual",
                op_type="ascend_laser_attention",
                layout=compute_layout,
            )
            return out
        else:  # SDPA
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            if q.shape[1] != k.shape[1]:
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=scale,
                    enable_gqa=True,
                )
            else:
                attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)

            return attn.transpose(1, 2)
