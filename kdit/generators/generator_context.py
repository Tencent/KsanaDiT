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

"""GeneratorInferContext — Generator.run() 的结构化输入上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from kdit.config import KsanaRuntimeConfig, KsanaSampleConfig
from kdit.config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
from kdit.config.video_control_config import KsanaVideoControlConfig
from kdit.utils.vace import KsanaVaceContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from kdit.models import KsanaDiffusionModel


@dataclass
class GeneratorInferContext:
    """Generator.run() 的输入上下文，收敛 14 个参数为结构化数据。

    将原本散落在 run() 签名中的参数分为四组：
    - 模型：diffusion_model
    - 输入 tensor：positive / negative / image_embeds / input_latent / noise_shape
    - 设备：device / offload_device
    - 配置：sample_config / runtime_config / cache_config / video_control / control_video_config / comfy_bar_callback
    """

    # 模型
    diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel] = None

    # 输入 tensor
    positive: torch.Tensor | tuple = None
    negative: torch.Tensor | tuple = None
    image_embeds: list[torch.Tensor] | None = None
    input_latent: torch.Tensor | None = None
    noise_shape: list[int] | None = None

    # 设备
    device: torch.device | None = None
    offload_device: torch.device | None = None

    # 配置
    sample_config: KsanaSampleConfig | None = None
    runtime_config: KsanaRuntimeConfig | None = None
    cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig] | None = None

    # 可选控制
    video_control: KsanaVideoControlConfig | None = None
    control_video_config: KsanaVaceContext | None = None
    comfy_bar_callback: Callable | None = None
