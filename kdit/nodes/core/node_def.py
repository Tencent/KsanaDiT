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

"""NodeDef — Node 实例的不可变定义 + NodeRef — Builder 返回的 Node 引用。"""

import itertools
from dataclasses import dataclass, field

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import IONodeType, NodeType
from kdit.nodes.core.pin_def import PinRef
from kdit.tensor.tensor_key import TensorKey

# ---------------------------------------------------------------------------
# 全局 node_id 分配器
# ---------------------------------------------------------------------------

_node_id_counter = itertools.count(1)


@dataclass(frozen=True)
class NodeDef:
    """Node 实例的不可变定义。

    Attributes:
        node_type: Node 的类型（IONodeType 或 InferNodeType）。
        model_key: 关联的模型（用于 Factory 查找 Node 类）。
        condition: 条件执行（ContextBuilder 上的方法名）。
        node_id: 全局自动分配的唯一 ID（不可由外部指定）。
    """

    node_type: NodeType
    model_key: ModelKey | None = None
    condition: str | None = None
    node_id: int = field(default_factory=lambda: next(_node_id_counter), init=False)

    @property
    def is_io(self) -> bool:
        """True 表示 IONode（Loader/Save/Read/Feed/Fetch），False 表示 InferNode。"""
        return isinstance(self.node_type, IONodeType)

    @property
    def is_loader(self) -> bool:
        """True 仅当 node_type == IONodeType.LOAD_MODEL。"""
        return self.node_type == IONodeType.LOAD_MODEL


# ---------------------------------------------------------------------------
# NodeRef — Builder 返回的 Node 引用，用于 connect 语法
# ---------------------------------------------------------------------------


class NodeRef:
    """NodeDef 的引用，用于 PipelineDefBuilder connect 语法。

    支持属性访问：node_ref.POSITIVE → PinRef(node_id, TensorKey.POSITIVE)
    """

    def __init__(self, node_def: NodeDef):
        self._node_def = node_def

    @property
    def node_id(self) -> int:
        return self._node_def.node_id

    @property
    def node_def(self) -> NodeDef:
        return self._node_def

    def __getattr__(self, name: str) -> PinRef:
        # vae_a.BASE_LATENT → PinRef(node_id, TensorKey.BASE_LATENT)
        if hasattr(TensorKey, name):
            return PinRef(self._node_def.node_id, getattr(TensorKey, name))
        if hasattr(ModelKey, name):
            return PinRef(self._node_def.node_id, getattr(ModelKey, name))
        raise AttributeError(f"Unknown pin: {name}")

    def __dir__(self):
        """IDE 自动补全支持。"""
        return list(TensorKey.__members__) + list(ModelKey.__members__)
