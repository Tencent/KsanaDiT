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

"""Generator 噪声与缓存创建函数 — 从 BaseGenerator 提取的无状态工具函数。"""

import random
import sys

import torch

from kdit.cache import create_hybrid_cache
from kdit.config import RuntimeConfig
from kdit.config.cache_config import KsanaCacheConfig
from kdit.models import ModelKey
from kdit.utils import log


def create_random_noise_latents(
    total_samples_num: int,
    noise_shape: tuple[int],
    runtime_config: RuntimeConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Generator]:
    """创建随机噪声 latents。

    Returns:
        (noise, seed_g) — noise shape: [bs, z_dim, f, h, w] (5D tensor for batch)
    """
    seed = (
        runtime_config.seed
        if runtime_config.seed is not None and runtime_config.seed >= 0
        else random.randint(0, sys.maxsize)
    )
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    latents_list = []
    for _ in range(total_samples_num):
        single_noise = torch.randn(
            *noise_shape,
            dtype=torch.float32,
            device=device,
            generator=seed_g,
        ).to(dtype)
        latents_list.append(single_noise)
    noise = torch.stack(latents_list, dim=0)
    log.info(f"create random noise_latents shape {noise.shape}, dtype:{noise.dtype}, device:{noise.device}")
    return noise, seed_g


def create_cache(cache_config: list[KsanaCacheConfig | None], model_key: ModelKey):
    """根据 cache_config 创建 hybrid cache 列表。"""
    if cache_config is None:
        return None
    cache = []
    for config in cache_config:
        if config is None:
            cache.append(None)
            continue
        cache.append(create_hybrid_cache(model_key, config))
    return cache
