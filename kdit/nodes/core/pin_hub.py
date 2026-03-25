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

from kdit.models.model_key import ModelKey
from kdit.models.model_pool import ModelPool
from kdit.models.model_pool_key import ModelPoolKey
from kdit.tensor.tensor_key import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey
from kdit.tensor.tensor_value import TensorData

from .node_def import NodeDef
from .node_types import Pins


class PinHub:
    """Node 运行时的数据访问器 — 由 Executor 根据 DAG 连线构建。

    每个 Node 实例拥有独立的 PinHub，被严格约束在 DAG 声明的范围内：
    - 读操作：只能读 input_pins 中声明的上游输出
    - 写操作：只能写自己 node_id 命名空间下的 key
    - get_model() / put_model() 支持无参调用（自动使用 node_def.model_key）
    """

    __slots__ = ("_node_def", "_tensor_pool", "_model_pool", "_model_mapping", "_tensor_mapping")

    def __init__(
        self,
        node_def: NodeDef,
        input_pins: Pins,
        tensor_pool: TensorPool,
        model_pool: ModelPool,
    ):
        self._node_def = node_def
        self._tensor_pool = tensor_pool
        self._model_pool = model_pool
        # 从 flat input_pins 中按值类型拆分 model / tensor 映射
        self._model_mapping: dict[ModelKey, ModelPoolKey] = {
            k: v for k, v in input_pins.items() if isinstance(v, ModelPoolKey)
        }
        self._tensor_mapping: dict[TensorKey, TensorPoolKey] = {
            k: v for k, v in input_pins.items() if isinstance(v, TensorPoolKey)
        }

    # ── Model 读写 ──

    def get_model(self, pin: ModelKey | None = None):
        """读取输入 model — 从 DAG 连线找到上游 ModelPoolKey。

        Args:
            pin: 指定 ModelKey。None 时自动使用 node_def.model_key（单 model 场景）。

        Raises:
            KeyError: pin 未在 input_pins 中声明。
        """
        if pin is None:
            pin = self._node_def.model_key
        pool_key = self._model_mapping.get(pin)
        if pool_key is None:
            raise KeyError(f"Model pin {pin} not connected for node {self._node_def.node_id}")
        return self._model_pool.get_model(pool_key)

    def put_model(self, model, pin: ModelKey | None = None) -> None:
        """写入输出 model — 自动用 node_id + pin 生成 ModelPoolKey。

        Args:
            model: 要写入的模型实例。
            pin: 指定 ModelKey。None 时自动使用 node_def.model_key。
        """
        if pin is None:
            pin = self._node_def.model_key
        pool_key = ModelPoolKey(self._node_def.node_id, pin)
        self._model_pool.update_model_with_key(pool_key, model)

    # ── Tensor 读写 ──

    def get_tensor(self, pin: TensorKey) -> TensorData | None:
        """读取输入 tensor — 从 DAG 连线找到上游 TensorPoolKey。

        返回 ``TensorData``（即 ``Tensor | list[Tensor]``），与
        ``InferNode._get_data`` 行为一致。未连线的可选输入返回 None。
        """
        pool_key = self._tensor_mapping.get(pin)
        if pool_key is None:
            return None  # 可选输入，未连线返回 None
        v = self._tensor_pool.get(pool_key)
        return v.data if v is not None else None

    def peek_tensor(self, pin: TensorKey) -> TensorData | None:
        """读取输入 tensor（不消费）— 从 DAG 连线找到上游 TensorPoolKey。

        当前 TensorPool.get() 本身不消费，peek 语义等价于 get。
        保留此方法以便未来 TensorPool 支持消费式 get 时平滑迁移。
        """
        pool_key = self._tensor_mapping.get(pin)
        if pool_key is None:
            return None
        v = self._tensor_pool.get(pool_key)
        return v.data if v is not None else None

    def put_tensor(self, pin: TensorKey, data: TensorData) -> None:
        """写入输出 tensor — 自动用 node_id + pin 生成 TensorPoolKey。"""
        pool_key = TensorPoolKey(self._node_def.node_id, pin)
        self._tensor_pool.put(pool_key, data)
