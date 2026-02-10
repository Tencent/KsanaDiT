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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ksana.accelerator import platform


class KsanaAttentionBackend(Enum):
    SAGE_ATTN = "sage_attention"
    FLASH_ATTN = "flash_attention"
    TORCH_SDPA = "torch_sdpa"
    RADIAL_SAGE_ATTN = "radial_sage_attention"
    SAGE_SLA = "sage_sla"
    LASER_ATTN = "laser_attention"

    @staticmethod
    def _get_npu_supported() -> list["KsanaAttentionBackend"]:
        return [
            KsanaAttentionBackend.TORCH_SDPA,
            KsanaAttentionBackend.LASER_ATTN,
        ]

    @staticmethod
    def _get_gpu_supported() -> list["KsanaAttentionBackend"]:
        return [
            KsanaAttentionBackend.SAGE_ATTN,
            KsanaAttentionBackend.FLASH_ATTN,
            KsanaAttentionBackend.TORCH_SDPA,
            KsanaAttentionBackend.RADIAL_SAGE_ATTN,
            KsanaAttentionBackend.SAGE_SLA,
        ]

    @staticmethod
    def get_supported_list(exclude: list[KsanaAttentionBackend] = None) -> list[str]:
        if exclude is None:
            exclude = []

        if platform.is_npu():
            supported_backends = KsanaAttentionBackend._get_npu_supported()
        else:
            supported_backends = KsanaAttentionBackend._get_gpu_supported()

        exclude_set = set(exclude)
        return [backend.value for backend in supported_backends if backend not in exclude_set]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaAttentionBackend.get_supported_list()
        elif isinstance(type, KsanaAttentionBackend):
            return True
        else:
            return False


# Attention backend: TORCH_SDPA for NPU, FLASH_ATTN for GPU
DEFAULT_ATTENTION_BACKEND = KsanaAttentionBackend.TORCH_SDPA if platform.is_npu() else KsanaAttentionBackend.FLASH_ATTN


@dataclass(frozen=True)
class KsanaAttentionConfig:
    backend: KsanaAttentionBackend | None = field(default=DEFAULT_ATTENTION_BACKEND)

    def __post_init__(self):
        if not KsanaAttentionBackend.support(self.backend):
            raise ValueError(
                f"attention_config {self.backend} not supported in {KsanaAttentionBackend.get_supported_list()}"
            )


@dataclass(frozen=True)
class KsanaRadialSageAttentionConfig(KsanaAttentionConfig):
    """Radial Sage Attention的配置"""

    backend: KsanaAttentionBackend = field(init=False, default=KsanaAttentionBackend.RADIAL_SAGE_ATTN)
    dense_blocks_num: int = field(default=0)
    dense_attn_steps: int = field(default=0)
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


@dataclass(frozen=True)
class KsanaSageSLAConfig(KsanaAttentionConfig):
    """Radial Sage Attention的配置"""

    backend: KsanaAttentionBackend = field(init=False, default=KsanaAttentionBackend.SAGE_SLA)
    dense_attention_config: KsanaAttentionConfig = field(
        default=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN)
    )
    topk: float = field(default=0.1)

    def __post_init__(self):
        if not (0 < self.topk < 1):
            raise ValueError(f"topk must be in range (0, 1), got {self.topk}")
