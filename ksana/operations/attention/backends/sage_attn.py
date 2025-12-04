from __future__ import annotations

import logging
from typing import Any, Optional

import torch

try:
    from sageattention import sageattn

    _SAGE_AVAILABLE = True
except ModuleNotFoundError:
    sageattn = None  # type: ignore[assignment]
    _SAGE_AVAILABLE = False

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata

logger = logging.getLogger(__name__)


class SageAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"

    @staticmethod
    def is_available() -> bool:
        return _SAGE_AVAILABLE

    @staticmethod
    def supports(head_size: int, dtype: torch.dtype) -> bool:
        return _SAGE_AVAILABLE

    @staticmethod
    def get_impl_cls() -> type["SageAttentionImpl"]:
        return SageAttentionImpl


class SageAttentionImpl(AttentionImpl[AttentionMetadata]):
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
        if not _SAGE_AVAILABLE:
            raise RuntimeError("SageAttention backend requested but 'sageattention' package is not installed. ")
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
        **kwargs: Any,
    ) -> torch.Tensor:
        if sageattn is None:
            raise RuntimeError("sageattention module missing at runtime.")
        original_dtype = query.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            target_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            )
            query = query.to(target_dtype)
            key = key.to(target_dtype)
            value = value.to(target_dtype)
        output = sageattn(
            query,
            key,
            value,
            tensor_layout="NHD",
            is_causal=self.causal,
        )
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        return output
