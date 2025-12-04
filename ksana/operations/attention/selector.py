from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Iterable, Optional, Sequence, Type

import torch

from .backends import (
    AttentionBackend,
    FlashAttentionBackend,
    SageAttentionBackend,
    SDPABackend,
)

logger = logging.getLogger(__name__)

KSANA_ATTENTION_BACKEND_ENV = "KSANA_ATTENTION_BACKEND"


class AttentionBackendEnum(str, Enum):
    FLASH_ATTN = "FLASH_ATTN"
    TORCH_SDPA = "TORCH_SDPA"
    SAGE_ATTN = "SAGE_ATTN"

    @classmethod
    def from_string(cls, value: str) -> "AttentionBackendEnum":
        """字符串转枚举的智能方法"""
        if value == "flash_attention":
            return cls.FLASH_ATTN
        elif value == "sage_attention":
            return cls.SAGE_ATTN
        return cls(value.upper())


def backend_name_to_enum(name: str | None) -> Optional[AttentionBackendEnum]:
    if name is None:
        return None
    try:
        return AttentionBackendEnum[name.upper()]
    except KeyError:
        return None


def _normalize_supported(
    supported_attention_backends: Optional[Sequence[AttentionBackendEnum]],
) -> tuple[AttentionBackendEnum, ...]:
    if supported_attention_backends:
        return tuple(dict.fromkeys(supported_attention_backends))
    return (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )


def _order_candidates(
    preferred: Optional[AttentionBackendEnum],
    supported: Iterable[AttentionBackendEnum],
) -> list[AttentionBackendEnum]:
    ordered: list[AttentionBackendEnum] = []
    if preferred is not None and preferred in supported:
        ordered.append(preferred)
    for backend in supported:
        if backend not in ordered:
            ordered.append(backend)
    return ordered


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    *,
    supported_attention_backends: Optional[Sequence[AttentionBackendEnum]] = None,
    forced_backend: Optional[AttentionBackendEnum] = None,
) -> Type[AttentionBackend]:
    supported = _normalize_supported(supported_attention_backends)
    env_backend = backend_name_to_enum(os.environ.get(KSANA_ATTENTION_BACKEND_ENV))
    preferred = forced_backend or env_backend
    candidates = _order_candidates(preferred, supported)

    for backend in candidates:
        if backend == AttentionBackendEnum.FLASH_ATTN:
            if FlashAttentionBackend.supports(head_size, dtype):
                logger.debug("Using FlashAttention backend for head_size=%s dtype=%s", head_size, dtype)
                return FlashAttentionBackend
            logger.debug(
                "FlashAttention backend unavailable (head_size=%s dtype=%s).",
                head_size,
                dtype,
            )
        elif backend == AttentionBackendEnum.SAGE_ATTN:
            if SageAttentionBackend.supports(head_size, dtype):
                logger.debug(
                    "Using Sage Attention backend for head_size=%s dtype=%s",
                    head_size,
                    dtype,
                )
                return SageAttentionBackend
            logger.debug(
                "Sage Attention backend unavailable (head_size=%s dtype=%s).",
                head_size,
                dtype,
            )
        elif backend == AttentionBackendEnum.TORCH_SDPA:
            logger.debug("Using torch SDPA backend for head_size=%s dtype=%s", head_size, dtype)
            return SDPABackend
        else:
            logger.error("Unsupported backend requested: %s", backend)

    raise RuntimeError(
        "No compatible attention backend available. "
        "Install flash-attn / sageattention or set KSANA_ATTENTION_BACKEND=TORCH_SDPA."
    )
