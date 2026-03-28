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

from dataclasses import dataclass, field

import torch

from kdit.config import RuntimeConfig, SampleConfig

from .device_info import DeviceInfo


@dataclass
class NodeContext:
    """Node 间传递的上下文，只包含可序列化数据。

    设计约束：
    - 不含任何 tensor（强制 __post_init__ 校验）
    - 不含 model_key（model_key 作为 NodeDef 的成员传入 engine.run_node）
    - 可安全跨 Ray 边界序列化
    """

    # 用户输入
    prompt: str | list[str] = None
    negative_prompt: str | list[str] = None
    img_path: str | list[str] | list[list[str]] = None

    # 配置
    sample_config: SampleConfig = None
    runtime_config: RuntimeConfig = None
    cache_config: list = None

    # 元数据（如 noise_shape、vace_config 等由上游 Node 或 Pipeline 写入）
    metadata: dict = field(default_factory=dict)

    # 设备信息 — Pipeline 层构建时为 None，Executor 在调用 Node 前注入
    device: DeviceInfo = None

    def __post_init__(self):
        """强制校验：不允许包含 tensor（包括 metadata dict 内部）。"""
        for field_name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                raise TypeError(
                    f"NodeContext.{field_name} is a Tensor! "
                    f"Use engine.feed_tensors() + TensorKey instead. "
                    f"NodeContext must be serializable for Ray dispatch."
                )
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        raise TypeError(
                            f"NodeContext.{field_name}[{k!r}] is a Tensor! "
                            f"Use engine.feed_tensors() + TensorKey instead. "
                            f"NodeContext must be serializable for Ray dispatch."
                        )
