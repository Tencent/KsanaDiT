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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from ..models.model_key import KsanaModelKey


class KsanaCache(ABC):
    def __init__(self, model_key: KsanaModelKey):
        self.model_key = model_key

    @abstractmethod
    def offload_to_cpu(self):
        pass

    @abstractmethod
    def show_cache_rate(self):
        pass


class KsanaStepCache(ABC):
    def __init__(self, model_key: KsanaModelKey, config):
        self.model_key = model_key
        self.input_config = config
        self.need_compile_cache = False
        self.timestep_start = None
        self.timestep_end = None

    @abstractmethod
    def valid_for(self, **kwargs) -> bool:
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        """Apply cache to current input and return output"""
        pass

    @abstractmethod
    def record_input_before_update(self, **kwargs):
        """record some inputs before blocks computing, update cache after blocks"""
        pass

    @abstractmethod
    def update_cache(self, **kwargs):
        """update cache for next time apply cache"""
        pass

    @abstractmethod
    def offload_to_cpu(self):
        pass

    @abstractmethod
    def show_cache_rate(self):
        pass


class KsanaBlockCache(ABC):
    def __init__(self, model_key: KsanaModelKey, config):
        self.model_key = model_key
        self.input_config = config

    @abstractmethod
    def __call__(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def offload_to_cpu(self):
        pass

    @abstractmethod
    def show_cache_rate(self):
        pass


@dataclass
class KsanaHybridCache(KsanaCache):
    name: str = field(default="hybrid_cache")
    model_key: KsanaModelKey = None
    step_cache: KsanaStepCache = None
    block_cache: KsanaBlockCache = None

    def __post_init__(self):
        if self.model_key is None:
            raise ValueError("KsanaHybridCache model_key must be set")
        self.name = f"{KsanaModelKey(self.model_key).name}_{self.name}"
        if self.step_cache is not None:
            self.name = f"{self.name}_{type(self.step_cache).__name__}"
        if self.block_cache is not None:
            self.name = f"{self.name}_{type(self.block_cache).__name__}"

    def offload_to_cpu(self):
        if self.step_cache is not None:
            self.step_cache.offload_to_cpu()
        if self.block_cache is not None:
            self.block_cache.offload_to_cpu()

    def show_cache_rate(self):
        if self.step_cache is not None:
            self.step_cache.show_cache_rate()
        if self.block_cache is not None:
            self.block_cache.show_cache_rate()
