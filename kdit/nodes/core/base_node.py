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

from kdit.models.model_pool import KsanaModelPool
from kdit.tensor import TensorKey, TensorPool

from .device_context import KsanaDeviceContext
from .node_context import KsanaNodeContext
from .node_types import KsanaDispatchPolicy


class KsanaLoadNode(ABC):
    """加载模型的 Node — 写入 model_pool。

    子类覆写 run() 实现具体加载逻辑。
    dispatch_policy 默认 ALL_ALL_ALL（每卡独立加载），子类可覆写。
    """

    dispatch_policy: KsanaDispatchPolicy = KsanaDispatchPolicy.ALL_ALL_ALL

    @abstractmethod
    def run(
        self,
        model_key,
        *,
        model_pool: KsanaModelPool,
        device_ctx: KsanaDeviceContext,
        **kwargs,
    ) -> None:
        """加载模型到 model_pool。"""


class KsanaInferNode(ABC):
    """前向推理的 Node — 读写 tensor_pool。

    子类覆写 run() 实现具体推理逻辑。
    所有结果必须写入 tensor_pool，run() 统一返回 None。
    外部通过 engine.get_tensor(key) 从 rank 0 的 tensor_pool 获取最终输出。

    约束：
    - run() 签名固定，禁止添加额外参数或 **kwargs
    - 输入 tensor 只能通过 tensor_pool.get()/peek() 获取
    - 输出 tensor 只能通过 tensor_pool.put() 写入
    - 额外配置通过 context.metadata 传递

    input_tensor_keys / output_tensor_keys 用于：
    - 声明 Node 的 tensor 输入/输出契约
    - R0_R0_BCAST 时 output_tensor_keys 指定需要 broadcast 的 key
    - 未来 executor 可据此做输入校验和 DAG 构建
    """

    dispatch_policy: KsanaDispatchPolicy = KsanaDispatchPolicy.ALL_ALL_ALL
    input_tensor_keys: list[str] = []
    output_tensor_keys: list[str] = []

    @staticmethod
    def _get_data(tensor_pool: TensorPool, key: TensorKey):
        """从 pool 取 TensorValue 并返回 .data，不存在返回 None。"""
        v = tensor_pool.get(key)
        return v.data if v is not None else None

    @abstractmethod
    def run(
        self,
        model_key,
        context: KsanaNodeContext,
        *,
        tensor_pool: TensorPool,
        model_pool: KsanaModelPool,
        device_ctx: KsanaDeviceContext,
    ) -> None:
        """前向推理 — 结果写入 tensor_pool，不返回值。"""
