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

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType


@dataclass(frozen=True)
class LoadPhase:
    """模型加载阶段 — 声明一个需要加载的模型。

    Attributes:
        model_key: 具体的模型 key（ModelKey 枚举值）。
    """

    model_key: ModelKey


@dataclass(frozen=True)
class InferPhase:
    """推理阶段 — 声明一个 InferNode 的执行。

    与 InferNodeFactory 的二级注册 (node_type, model_key) 对齐。

    Attributes:
        node_type: InferNode 类型枚举。
        model_key: 关联的 ModelKey（与 LoadPhase 对应），
                   SaveNode 等无模型 Node 为 None。
        condition: ContextBuilder 上的条件方法名，
                   为 None 时无条件执行。
    """

    node_type: InferNodeType
    model_key: ModelKey | None = None
    condition: str | None = None
