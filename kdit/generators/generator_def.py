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

"""GeneratorDef — Generator 的声明式定义 + 全局注册表。

与 PipelineDef 对齐：frozen dataclass + 注册表模式。
"""

from dataclasses import dataclass, field

from kdit.models.model_key import ModelKey

from .handlers.denoise_handler import DenoiseHandler
from .handlers.latent_handler import LatentHandler
from .handlers.text_handler import TextHandler


@dataclass(frozen=True)
class GeneratorDef:
    """Generator 的声明式定义 — 不可变，一次构建终身复用。

    与 PipelineDef 对齐：frozen dataclass + 注册表。
    只有 3 个 Handler 字段 + 1 个 model_key。
    """

    model_key: ModelKey
    text_handler: TextHandler = field(default_factory=TextHandler)
    latent_handler: LatentHandler = field(default_factory=LatentHandler)
    denoise_handler: DenoiseHandler = field(default_factory=DenoiseHandler)


# ── 注册表 ──

_GENERATOR_DEF_REGISTRY: dict[ModelKey, GeneratorDef] = {}


def register_generator_def(generator_def: GeneratorDef) -> GeneratorDef:
    """注册 GeneratorDef 到全局注册表。"""
    key = generator_def.model_key
    if key in _GENERATOR_DEF_REGISTRY:
        raise ValueError(f"GeneratorDef for {key} already registered")
    _GENERATOR_DEF_REGISTRY[key] = generator_def
    return generator_def


def get_generator_def(model_key: ModelKey) -> GeneratorDef:
    """从注册表获取 GeneratorDef。"""
    if model_key not in _GENERATOR_DEF_REGISTRY:
        raise KeyError(f"No GeneratorDef registered for {model_key}")
    return _GENERATOR_DEF_REGISTRY[model_key]


def reset_generator_def_registry():
    """清空注册表 — 仅用于测试。"""
    _GENERATOR_DEF_REGISTRY.clear()
