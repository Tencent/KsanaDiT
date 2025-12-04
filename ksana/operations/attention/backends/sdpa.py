from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl[AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        **extra_impl_args: Any,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
        *,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
        dropout_p: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        **_: Any,
    ) -> torch.Tensor:
        drop = self.dropout if dropout_p is None else dropout_p
        scale = self.softmax_scale if softmax_scale is None else softmax_scale

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if q.shape[1] != k.shape[1]:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=drop,
                is_causal=self.causal,
                scale=scale,
                enable_gqa=True,
            )
        else:
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=self.causal, scale=scale)

        return attn.transpose(1, 2)
