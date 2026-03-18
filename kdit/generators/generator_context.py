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

from collections.abc import Callable
from dataclasses import dataclass

import torch

from kdit.config import RuntimeConfig, SampleConfig
from kdit.config.cache_config import CacheConfig, HybridCacheConfig
from kdit.config.video_control_config import KsanaVideoControlConfig
from kdit.models import KsanaDiffusionModel
from kdit.utils.vace import VaceConfig

# ── image_embeds 类型别名 ─────────────────────────────────────────────────
# ImageEmbeds: 单个 Tensor，shape[0] = batch 维度。
#   适用于 WAN I2V（单张首帧）、ComfyUI 单 prompt 等场景。
ImageEmbeds = torch.Tensor

# MultiPromptImageEmbeds: list[Tensor]，list 长度 = prompt 数量，
#   每个 Tensor.shape[0] = 该 prompt 对应的参考图数量（num_refs）。
#   适用于 Qwen Edit Pipeline（每个 prompt 可有不同参考图组）。
MultiPromptImageEmbeds = list[torch.Tensor]


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
    image_embeds: ImageEmbeds | MultiPromptImageEmbeds | None = None
    input_latent: torch.Tensor | None = None
    noise_shape: list[int] | None = None

    # 设备
    device: torch.device | None = None
    offload_device: torch.device | None = None

    # 配置
    sample_config: SampleConfig | None = None
    runtime_config: RuntimeConfig | None = None
    cache_config: list[CacheConfig | HybridCacheConfig] | None = None

    # 可选控制
    video_control: KsanaVideoControlConfig | None = None
    control_video_config: VaceConfig | None = None
    comfy_bar_callback: Callable | None = None
