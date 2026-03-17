# Copyright 2026 Tencent
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

"""GenerateInputs — 所有 Pipeline 共有的最小公共输入集。

Pipeline 特有的输入由 ContextBuilder.prepare_generate_inputs() 管理，
存储在 ContextBuilder._extra 中。
"""

from dataclasses import dataclass

from kdit.config import RuntimeConfig, SampleConfig
from kdit.config.cache_config import CacheConfig, HybridCacheConfig


@dataclass
class GenerateInputs:
    """所有 Pipeline 共有的输入 — 最小公共集。

    Attributes:
        prompt: 正向提示词（单条或列表）。
        prompt_negative: 负向提示词（可选）。
        num_prompts: prompt 数量。
        sample_config: 采样配置（已校验）。
        runtime_config: 运行时配置（已校验）。
        cache_config: 缓存配置列表（可选，已校验）。
        has_lora: 是否使用了 LoRA。
    """

    prompt: str | list[str]
    prompt_negative: str | list[str] | None
    num_prompts: int
    sample_config: SampleConfig
    runtime_config: RuntimeConfig
    cache_config: list[CacheConfig | HybridCacheConfig] | None
    has_lora: bool
