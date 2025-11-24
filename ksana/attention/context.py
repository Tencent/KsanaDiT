from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from .backends.abstract import AttentionMetadata


@dataclass
class ForwardContext:
    current_timestep: Optional[int] = None
    attn_metadata: Optional[AttentionMetadata] = None


_CURRENT_CONTEXT: ForwardContext | None = None


def get_forward_context() -> ForwardContext | None:
    """Return the currently active forward context if one is set."""
    return _CURRENT_CONTEXT


@contextmanager
def set_forward_context(
    current_timestep: Optional[int] = None,
    attn_metadata: Optional[AttentionMetadata] = None,
) -> None:
    """Context manager used to set and restore the global forward context."""
    global _CURRENT_CONTEXT
    previous_context = _CURRENT_CONTEXT
    _CURRENT_CONTEXT = ForwardContext(
        current_timestep=current_timestep,
        attn_metadata=attn_metadata,
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT = previous_context
