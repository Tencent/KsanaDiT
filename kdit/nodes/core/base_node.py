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

from kdit.models.model_key import ModelKey
from kdit.tensor import TensorKey

from .node_context import NodeContext
from .node_types import NodeDispatchPolicy
from .pin_hub import PinHub


class LoaderNode(ABC):
    """加载模型的 Node — 通过 PinHub 写入 ModelPool。

    子类覆写 run() 实现具体加载逻辑。
    dispatch_policy 默认 ALL_ALL_ALL（每卡独立加载），子类可覆写。

    run() 签名统一为 ``(self, pins: PinHub, *, context: NodeContext)``：
    - ``pins.put_model(model_key, model)`` 写入加载好的模型
    - ``context.metadata`` 获取加载参数（model_path / model_config 等）
    - ``context.device`` 获取设备信息（由 Executor 注入）
    """

    dispatch_policy: NodeDispatchPolicy = NodeDispatchPolicy.ALL_ALL_ALL

    # output_model_pins 由 Factory 注册时的 ModelKey 自动填充，不需要手动写
    output_model_pins: list[ModelKey] = []

    @abstractmethod
    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        """加载模型 — 通过 pins.put_model() 写入，加载参数从 context.metadata 获取。"""


class InferNode(ABC):
    """前向推理的 Node — 通过 PinHub 读写数据。

    子类覆写 run() 实现具体推理逻辑。
    所有结果必须通过 pins.put_tensor() 写入，run() 统一返回 None。
    外部通过 engine.get_tensor(key) 从 rank 0 的 tensor_pool 获取最终输出。

    run() 签名统一为 ``(self, pins: PinHub, *, context: NodeContext)``：
    - ``pins.get_model(model_key)`` 读取模型
    - ``pins.get_tensor(tensor_key)`` 读取输入 tensor
    - ``pins.put_tensor(tensor_key, data)`` 写入输出 tensor
    - ``context.device`` 获取设备信息（由 Executor 注入）
    - ``context.metadata`` 获取额外配置

    Pin 声明：
    - ``input_model_pins``: 由 Factory.create() 自动注入，Node 内通过 ``self.input_model_pins[0]`` 获取
    - ``input_tensor_pins`` / ``output_tensor_pins``: 子类声明 tensor 输入/输出端口
    - R0_R0_BCAST 时 ``output_tensor_pins`` 指定需要 broadcast 的 key
    """

    dispatch_policy: NodeDispatchPolicy = NodeDispatchPolicy.ALL_ALL_ALL

    # input_model_pins 由 Factory.create() 自动注入，不需要手动写
    input_model_pins: list[ModelKey] = []
    input_tensor_pins: list[TensorKey] = []
    output_tensor_pins: list[TensorKey] = []

    @abstractmethod
    def run(self, pins: PinHub, *, context: NodeContext) -> None:
        """前向推理 — 通过 pins 读写数据，通过 context 获取配置和设备信息。"""
