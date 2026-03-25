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


from dataclasses import dataclass

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import IONodeType, NodeType


@dataclass(frozen=True)
class NodeDef:
    """DAG 中一个 Node 实例的定义。

    Attributes:
        node_id: Builder 自动分配的唯一 ID。
        node_type: Node 的类型（IONodeType 或 InferNodeType）。
        model_key: 关联的模型（用于 Factory 查找 Node 类）。
        condition: 条件执行（ContextBuilder 上的方法名）。
    """

    node_id: int
    node_type: NodeType
    model_key: ModelKey | None = None
    condition: str | None = None

    @property
    def is_io(self) -> bool:
        """True 表示 IONode（Loader/Save/Read），False 表示 InferNode。"""
        return isinstance(self.node_type, IONodeType)

    @property
    def is_loader(self) -> bool:
        """向后兼容别名 — 等价于 is_io。"""
        return self.is_io
