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

"""Pipeline 声明式定义 — LoadPhase, InferPhase"""


from dataclasses import dataclass

from kdit.models.model_key import KsanaModelKey
from kdit.nodes.core.node_types import KsanaInferNodeType


@dataclass(frozen=True)
class LoadPhase:
    """模型加载阶段 — 声明一个需要加载的模型。

    Attributes:
        model_role: 角色名（如 "text_encoder", "diffusion", "vae")
                    在 InferPhase 中通过同名引用。
        model_key: 具体的模型 key。
    """

    model_role: str
    model_key: KsanaModelKey


@dataclass(frozen=True)
class InferPhase:
    """推理阶段 — 声明一个 InferNode 的执行。

    Attributes:
        node_type: InferNode 类型枚举。
        model_role: 关联的 model_role（与 LoadPhase 对应），
                    SaveNode 等无模型 Node 为 None。
        condition: ContextBuilder 上的条件方法名，
                   为 None 时无条件执行。
    """

    node_type: KsanaInferNodeType
    model_role: str | None = None
    condition: str | None = None
