from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class KsanaAttentionBackend(Enum):
    SAGE_ATTN = "sage_attention"
    FLASH_ATTN = "flash_attention"
    TORCH_SDPA = "torch_sdpa"
    RADIAL_SAGE_ATTN = "radial_sage_attention"

    @staticmethod
    def get_supported_list(exclude: list[KsanaAttentionBackend] = None) -> list[str]:
        if exclude is None:
            exclude = []
        return [b.value for b in KsanaAttentionBackend if b not in exclude]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaAttentionBackend.get_supported_list()
        elif isinstance(type, KsanaAttentionBackend):
            return True
        else:
            return False


@dataclass(frozen=True)
class KsanaAttentionConfig:
    backend: KsanaAttentionBackend | None = field(default=KsanaAttentionBackend.FLASH_ATTN)

    def __post_init__(self):
        if not KsanaAttentionBackend.support(self.backend):
            raise ValueError(
                f"attention_config {self.backend} not supported in {KsanaAttentionBackend.get_supported_list()}"
            )


@dataclass(frozen=True)
class KsanaRadialSageAttentionConfig(KsanaAttentionConfig):
    """Radial Sage Attention的配置"""

    backend: KsanaAttentionBackend = field(init=False, default=KsanaAttentionBackend.RADIAL_SAGE_ATTN)
    dense_blocks_num: int = field(default=1)
    dense_attn_steps: int = field(default=1)
    decay_factor: float = field(default=0.02)
    block_size: int = field(default=128)
    dense_attention_config: KsanaAttentionConfig = field(
        default=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN)
    )

    def __post_init__(self):
        if self.block_size not in [64, 128]:
            raise ValueError(f"block_size must be 64 or 128, got {self.block_size}")

        if not (0 < self.decay_factor < 1):
            raise ValueError(f"decay_factor must be in range (0, 1), got {self.decay_factor}")
