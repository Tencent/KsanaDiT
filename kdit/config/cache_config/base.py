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

from dataclasses import dataclass, field


@dataclass
class KsanaCacheConfig:
    name: str = field(default="KsanaCache")
    offload: bool = field(default=False)


@dataclass
class KsanaBlockCacheConfig(KsanaCacheConfig):
    name: str = field(default="KsanaBlockCache")


@dataclass
class KsanaStepCacheConfig(KsanaCacheConfig):
    name: str = field(default="KsanaStepCache")


@dataclass
class KsanaHybridCacheConfig:
    name: str = field(default="KsanaHybridCache")
    step_cache: KsanaStepCacheConfig | None = None
    block_cache: KsanaBlockCacheConfig | None = None

    def __post_init__(self):
        if self.step_cache is None and self.block_cache is None:
            raise ValueError("KsanaHybridCacheConfig must have step_cache or block_cache")


def warp_as_hybrid_cache(
    cache_config: KsanaCacheConfig,
) -> KsanaHybridCacheConfig:
    if isinstance(cache_config, KsanaHybridCacheConfig):
        return cache_config
    if cache_config is None or not isinstance(cache_config, KsanaCacheConfig):
        raise ValueError(f"cache_config must be provided, but got {cache_config}")
    if isinstance(cache_config, KsanaBlockCacheConfig):
        return KsanaHybridCacheConfig(name=cache_config.name, block_cache=cache_config)
    elif isinstance(cache_config, KsanaStepCacheConfig):
        return KsanaHybridCacheConfig(name=cache_config.name, step_cache=cache_config)
    else:
        raise ValueError(
            f"cache_config must be KsanaBlockCacheConfig or KsanaStepCacheConfig, but got {type(cache_config)}"
        )
