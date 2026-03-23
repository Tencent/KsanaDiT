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
from kdit.tensor.tensor_key import TensorKey


@dataclass(frozen=True)
class PinRef:
    """connect 时引用某个 Node 的某个 pin。

    只有两个字段：node_id + pin 枚举。
    pin 的类型（TensorKey 或 ModelKey）隐含了是 tensor 连线还是 model 连线。
    """

    node_id: int
    pin: TensorKey | ModelKey


class NodeRef:
    """add_loader/add_infer 返回的 Node 引用，用于 connect。

    支持属性访问：node_ref.POSITIVE → PinRef(node_id, TensorKey.POSITIVE)
    """

    def __init__(self, node_id: int):
        self._node_id = node_id

    @property
    def node_id(self) -> int:
        return self._node_id

    def __getattr__(self, name: str) -> PinRef:
        # vae_a.BASE_LATENT → PinRef(node_id, TensorKey.BASE_LATENT)
        if hasattr(TensorKey, name):
            return PinRef(self._node_id, getattr(TensorKey, name))
        if hasattr(ModelKey, name):
            return PinRef(self._node_id, getattr(ModelKey, name))
        raise AttributeError(f"Unknown pin: {name}")

    def __dir__(self):
        """IDE 自动补全支持。"""
        return list(TensorKey.__members__) + list(ModelKey.__members__)
