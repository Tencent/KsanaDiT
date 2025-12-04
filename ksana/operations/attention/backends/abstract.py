from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Generic, Optional, Protocol, TypeVar

import torch


class AttentionBackend(ABC):
    """Abstract base class for attention backend definitions."""

    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return IdentityAttentionMetadataBuilder


@dataclass
class AttentionMetadata:
    """Metadata container shared with attention backends."""

    current_timestep: Optional[int] = None

    def as_dict(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        skip_fields = skip_fields or set()
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name not in skip_fields}


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class AttentionMetadataBuilder(ABC, Generic[TMetadata]):
    """Factory base class for backend specific metadata objects."""

    def prepare(self) -> None:
        """Hook executed before building metadata for a new batch."""

    @abstractmethod
    def build(self, **kwargs: Any) -> Optional[TMetadata]:
        """Return per-forward metadata object."""


class IdentityAttentionMetadataBuilder(AttentionMetadataBuilder[AttentionMetadata]):
    """Metadata builder used for backends that do not require metadata."""

    def build(self, **kwargs: Any) -> Optional[AttentionMetadata]:
        current_timestep = kwargs.get("current_timestep")
        if current_timestep is None:
            return None
        return AttentionMetadata(current_timestep=current_timestep)


class AttentionLayer(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
    ) -> torch.Tensor: ...


class AttentionImpl(ABC, Generic[TMetadata]):
    """Runtime implementation executed by an attention backend."""

    @abstractmethod
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
        raise NotImplementedError

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: Optional[TMetadata],
    ) -> torch.Tensor:
        return qkv

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: Optional[TMetadata],
    ) -> torch.Tensor:
        return output

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[TMetadata],
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
