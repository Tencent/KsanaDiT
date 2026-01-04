from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch


class KsanaAttentionBackend(Enum):
    SAGE_ATTN = "sage_attention"
    FLASH_ATTN = "flash_attention"
    TORCH_SDPA = "torch_sdpa"

    @staticmethod
    def get_supported_list() -> list[str]:
        return [b.value for b in KsanaAttentionBackend]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaAttentionBackend.get_supported_list()
        elif isinstance(type, KsanaAttentionBackend):
            return True
        else:
            return False


class KsanaAttentionBackendImpl(ABC):
    """Implementation of attention backend."""

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        **extra_impl_args,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def type() -> KsanaAttentionBackend:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def supports(**kwargs) -> bool:
        raise NotImplementedError

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
    ) -> torch.Tensor:
        return qkv

    def postprocess_output(
        self,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
