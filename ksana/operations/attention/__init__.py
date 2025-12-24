from __future__ import annotations

from typing import Any
from .backends.abstract import AttentionBackend, AttentionMetadata, AttentionMetadataBuilder
from .attention_op import LocalAttentionOp
from .selector import AttentionBackendEnum, get_attn_backend
from .context import ForwardContext, get_forward_context, set_forward_context
from .pick_attn import pick_attn_op
from .backends.flash_attn import FlashAttentionBackend
from .backends.sage_attn import SageAttentionBackend

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "AttentionMetadataBuilder",
    "LocalAttentionOp",
    "AttentionBackendEnum",
    "get_attn_backend",
    "ForwardContext",
    "get_forward_context",
    "set_forward_context",
    "pick_attn_op",
    "FlashAttentionBackend",
    "SageAttentionBackend",
]

_ATTN_OP_CACHE: dict[tuple[int, int, int, AttentionBackendEnum], LocalAttentionOp] = {}


def _get_attn_op(q, k, *, backend: AttentionBackendEnum) -> LocalAttentionOp:
    key = (q.size(2), q.size(3), k.size(2), backend)
    op = _ATTN_OP_CACHE.get(key)
    if op is None:
        op = LocalAttentionOp(
            num_heads=key[0],
            head_size=key[1],
            num_kv_heads=key[2],
            attn_backend=backend,
        )
        _ATTN_OP_CACHE[key] = op
    return op


def attn_func(q, k, v, *, attn_backend: str | AttentionBackendEnum = "flash_attention", **kwargs: Any):
    backend = AttentionBackendEnum.from_string(attn_backend) if isinstance(attn_backend, str) else attn_backend
    op = _get_attn_op(q, k, backend=backend)
    return op(q, k, v, **kwargs)
