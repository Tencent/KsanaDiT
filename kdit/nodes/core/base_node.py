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

from abc import ABC, abstractmethod

from .node_context import NodeContext
from .node_types import NodeDispatchPolicy, PinDef
from .pin_hub import PinHub


class IONode(ABC):
    """IO Node — 加载模型 / 保存文件 / 读取文件。

    子类覆写 run() 实现具体逻辑。
    dispatch_policy 默认 ALL_ALL_ALL（每卡独立执行），子类可覆写。

    run() 签名统一为 ``(self, pins: PinHub, *, context: NodeContext)``：
    - ``pins.put_model(model_key, model)`` 写入加载好的模型
    - ``context.metadata`` 获取加载参数（model_path / model_config 等）
    - ``context.device`` 获取设备信息（由 Executor 注入）

    Def 声明：
    - ``input_defs`` / ``output_defs``: 声明 tensor 端口（``list[PinDef]``）
    - model 端口由 ``NodeDef.model_key`` 隐含，不在 Def 中声明
    """

    dispatch_policy: NodeDispatchPolicy = NodeDispatchPolicy.ALL_ALL_ALL

    # Def 声明 — tensor 端口
    input_defs: list[PinDef] = []
    output_defs: list[PinDef] = []

    @abstractmethod
    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        """执行 IO 操作 — 通过 pins 读写数据，参数从 context.metadata 获取。"""


class InferNode(ABC):
    """推理 Node — 前向计算。

    子类覆写 run() 实现具体推理逻辑。
    所有结果必须通过 pins.put_tensor() 写入，run() 统一返回 None。
    外部通过 engine.get_tensor(key) 从 rank 0 的 tensor_pool 获取最终输出。

    run() 签名统一为 ``(self, pins: PinHub, *, context: NodeContext)``：
    - ``pins.get_model()`` 读取模型（无参时自动从 node_def.model_key 获取）
    - ``pins.get_tensor(tensor_key)`` 读取输入 tensor
    - ``pins.put_tensor(tensor_key, data)`` 写入输出 tensor
    - ``context.device`` 获取设备信息（由 Executor 注入）
    - ``context.metadata`` 获取额外配置

    Def 声明：
    - ``input_defs`` / ``output_defs``: 声明 tensor 端口（``list[PinDef]``）
    - model 端口由 ``NodeDef.model_key`` 隐含，不在 Def 中声明
    - R0_R0_BCAST 时 ``output_defs`` 中的 TensorKey 指定需要 broadcast 的 key
    """

    dispatch_policy: NodeDispatchPolicy = NodeDispatchPolicy.ALL_ALL_ALL

    # Def 声明 — tensor 端口
    input_defs: list[PinDef] = []
    output_defs: list[PinDef] = []

    @abstractmethod
    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        """前向推理 — 通过 pins 读写数据，通过 context 获取配置和设备信息。"""
