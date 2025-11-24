from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .backends import AttentionBackend, AttentionImpl, AttentionMetadata
from .context import get_forward_context
from .selector import AttentionBackendEnum, get_attn_backend


class LocalAttentionOp(nn.Module):
    """Lightweight attention wrapper that delegates execution to registered backends."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        *,
        num_kv_heads: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        supported_attention_backends: Optional[tuple[AttentionBackendEnum, ...]] = None,
        forced_backend: Optional[AttentionBackendEnum] = None,
        **extra_impl_args: Any,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = softmax_scale if softmax_scale is not None else head_size**-0.5
        self.causal = causal
        self.supported_attention_backends = supported_attention_backends
        self.forced_backend = forced_backend
        self.extra_impl_args = extra_impl_args

        self._backend_cls: type[AttentionBackend] | None = None
        self._attn_impl: AttentionImpl | None = None
        self._attn_impl_dtype: torch.dtype | None = None

    @property
    def backend_name(self) -> Optional[str]:
        return self._backend_cls.get_name() if self._backend_cls else None

    def _ensure_impl(self, dtype: torch.dtype) -> None:
        if self._attn_impl is not None and dtype == self._attn_impl_dtype:
            return

        backend_cls = get_attn_backend(
            self.head_size,
            dtype,
            supported_attention_backends=self.supported_attention_backends,
            forced_backend=self.forced_backend,
        )
        impl_cls = backend_cls.get_impl_cls()
        self._attn_impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            prefix="local_attention",
            **self.extra_impl_args,
        )
        self._backend_cls = backend_cls
        self._attn_impl_dtype = dtype

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **backend_kwargs: Any,
    ) -> torch.Tensor:
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("LocalAttentionOp expects tensors shaped [B, L, H, D].")

        self._ensure_impl(query.dtype)
        assert self._attn_impl is not None

        ctx = get_forward_context()
        attn_metadata: AttentionMetadata | None = None
        if ctx is not None:
            attn_metadata = ctx.attn_metadata

        return self._attn_impl.forward(
            query,
            key,
            value,
            attn_metadata,
            **backend_kwargs,
        )
