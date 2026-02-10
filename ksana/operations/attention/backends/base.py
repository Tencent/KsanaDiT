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

from abc import ABC, abstractmethod

import torch

from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig
from ksana.utils.logger import log

_ATTN_BACKEND_TO_IMPL = None


def get_backend_mapping():
    global _ATTN_BACKEND_TO_IMPL

    if _ATTN_BACKEND_TO_IMPL is None:
        # lazy import to avoid circular import
        from .flash_attn import FlashAttentionImpl
        from .laser_attn import LaserAttentionImpl
        from .radial_sage_attn.radial_sage_attn import RadialSageAttentionImpl
        from .sage_attn import SageAttentionImpl
        from .sage_sla import SageSLAAttentionImpl
        from .sdpa import SDPAImpl

        _ATTN_BACKEND_TO_IMPL = {
            FlashAttentionImpl.type(): FlashAttentionImpl,
            SageAttentionImpl.type(): SageAttentionImpl,
            SDPAImpl.type(): SDPAImpl,
            LaserAttentionImpl.type(): LaserAttentionImpl,
            SageSLAAttentionImpl.type(): SageSLAAttentionImpl,
            RadialSageAttentionImpl.type(): RadialSageAttentionImpl,
        }

    return _ATTN_BACKEND_TO_IMPL


def get_attention_backend_impl(attention_config: KsanaAttentionConfig, **kwargs) -> KsanaAttentionBackendImpl:
    backend_mapping = get_backend_mapping()
    attn_backend = attention_config.backend
    if not KsanaAttentionBackend.support(attn_backend):
        raise ValueError(
            f"attn_backend:{attn_backend} is not in supported_list:{ KsanaAttentionBackend.get_supported_list()}"
        )
    # input attn backend at first
    for backend_type in [attn_backend] + KsanaAttentionBackend.get_supported_list():
        backend_type = KsanaAttentionBackend(backend_type)
        backend_impl = backend_mapping.get(backend_type, None)
        if backend_impl is None:
            raise ValueError(f"{backend_type} not in {backend_mapping.keys()}")
        if backend_impl.supports(**kwargs):
            log.debug(f"Using {backend_impl.type()} backend for {kwargs}")
            return backend_impl
        else:
            log.debug(f"{backend_impl.type()} backend unavailable for {kwargs}")
            continue

    raise RuntimeError(
        f"No compatible attention({KsanaAttentionBackend.get_supported_list()}) backend available for {kwargs}. "
    )


class KsanaAttentionBackendImpl(ABC):
    """Implementation of attention backend."""

    @abstractmethod
    def __init__(
        self,
        attention_config: KsanaAttentionConfig,
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

    def check_config(self):
        if KsanaAttentionBackend(self.attention_config.backend) != self.type():
            raise ValueError(
                f"Attention config {self.attention_config.backend} does not match implementation type {self.type()}"
            )

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
