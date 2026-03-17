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
from __future__ import annotations

from enum import Enum, auto, unique


@unique
class PipelineKey(Enum):
    """标识一条完整的推理流水线（Pipeline 级别语义）。

    零依赖枚举 — 可被任意子包安全导入，不会触发 kdit/__init__.py 的重量级导入链。
    """

    Wan2_2_T2V_14B = auto()
    Wan2_2_I2V_14B = auto()
    Wan2_2_TI2V_5B = auto()
    Wan2_1_VACE_14B = auto()
    QwenImage_T2I = auto()
    QwenImage_Edit = auto()

    # TODO: ToBe remove
    def is_i2v_type(self) -> bool:
        return self in (PipelineKey.Wan2_2_I2V_14B,)

    def is_image_type(self) -> bool:
        return self in (PipelineKey.QwenImage_T2I, PipelineKey.QwenImage_Edit)


def get_pipeline_key_from_path(model_path: str | list[str]) -> PipelineKey:
    """从 diffusion 模型路径推导 PipelineKey。

    内部复用 get_model_key_from_path() 得到 diffusion ModelKey，
    再通过同名映射转换为 PipelineKey。
    """
    from kdit.models.model_key import DIFFUSION_KEYS, get_model_key_from_path

    model_key = get_model_key_from_path(model_path)
    if model_key not in DIFFUSION_KEYS:
        raise ValueError(
            f"get_pipeline_key_from_path() expects a diffusion model path, "
            f"but got ModelKey.{model_key.name} which is not a diffusion key."
        )
    return PipelineKey[model_key.name]
