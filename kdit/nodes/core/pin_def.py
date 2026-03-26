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

"""Pin 类型定义 — PinDef, PinPoolKey, Pins, PinRef。

Pin 是 Node 的输入/输出端口。本模块定义了 Pin 的静态声明类型（PinDef）、
运行时 Pool Key 类型（PinPoolKey）、DAG 连线映射（Pins）、以及连线引用（PinRef）。
"""

from __future__ import annotations

from dataclasses import dataclass

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.tensor.tensor_key import TensorKey
from kdit.tensor.tensor_pool_key import TensorPoolKey

# ---------------------------------------------------------------------------
# 类型别名 — Def（静态声明）与 Pin（运行时映射）
# ---------------------------------------------------------------------------

#: Node 声明的端口类型（TensorKey 或 ModelKey）
PinDef = TensorKey | ModelKey

#: 运行时 pool 中的实际 key（TensorPoolKey 或 ModelPoolKey）
PinPoolKey = TensorPoolKey | ModelPoolKey

#: DAG 连线映射：PinDef → PinPoolKey
Pins = dict[PinDef, PinPoolKey]


# ---------------------------------------------------------------------------
# PinRef — 连线时引用某个 Node 的某个 Pin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinRef:
    """connect 时引用某个 Node 的某个 pin。

    只有两个字段：node_id + pin 枚举。
    pin 的类型（TensorKey 或 ModelKey）隐含了是 tensor 连线还是 model 连线。

    支持 ``>>`` 操作符用于声明连线::

        src_node.POSITIVE >> dst_node.POSITIVE
    """

    node_id: int
    pin: TensorKey | ModelKey

    def __rshift__(self, other: PinRef) -> tuple[PinRef, PinRef]:
        """``src >> dst`` — 返回 (src, dst) 元组，供 connect() 使用。"""
        if not isinstance(other, PinRef):
            return NotImplemented
        return (self, other)
