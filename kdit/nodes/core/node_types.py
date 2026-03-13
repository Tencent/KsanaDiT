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

from enum import Enum, auto


class KsanaDispatchPolicy(Enum):
    """Node 的多卡调度策略 — 三维度拼接命名：input_exec_output。

    每个策略隐含三个维度：
    - 输入要求：Node 期望 tensor_pool 中的输入在哪些卡上可用
    - 执行范围：run() 在哪些卡上被调用
    - 输出行为：结果如何同步到其他卡

    | Policy              | 输入要求      | 执行范围  | 输出行为          | 典型场景                |
    |---------------------|-------------|----------|------------------|----------------------|
    | ALL_ALL_ALL         | 所有卡都有    | 所有卡    | 各卡独立持有       | TextEncode, Generator |
    | R0_R0_BCAST         | rank0 有即可  | 仅 rank0 | broadcast 到所有卡 | VAEEncode             |
    | ALL_R0_R0           | 所有卡都有    | 仅 rank0 | 仅 rank0 持有     | VAEDecode             |
    """

    ALL_ALL_ALL = auto()
    R0_R0_BCAST = auto()
    ALL_R0_R0 = auto()


class KsanaInferNodeType(Enum):
    """Infer Node 类型枚举，用于 KsanaInferNodeFactory 的二级注册键。"""

    TEXT_ENCODE = auto()
    VAE_ENCODE_SPATIAL = auto()
    VAE_ENCODE_IMAGES = auto()
    VAE_DECODE = auto()
    GENERATE = auto()
