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

"""ContextBuilder — 为 Pipeline 的每个 InferPhase 构建 NodeContext 和准备 tensor。

每个 Pipeline 变体实现自己的 ContextBuilder 子类。

生命周期：
1. prepare_generate_inputs(base_inputs, **kwargs) — 一次性：提取 Pipeline 特有输入
2. 对每个 InferPhase:
   a. check_condition(name, inputs) — 是否跳过
   b. prepare_tensors(phase, inputs) — 准备 tensor -> put 到 pool
   c. build_context(phase, inputs) — 构建 KsanaNodeContext
3. post_process(output, inputs) — 输出后处理
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from kdit.nodes.core.node_context import KsanaNodeContext
from kdit.tensor import TensorKey

from .generate_inputs import GenerateInputs
from .pipeline_phase import InferPhase


class ContextBuilder(ABC):
    """为 Pipeline 的每个 InferPhase 构建 NodeContext 和准备 tensor。

    每个 Pipeline 变体实现自己的 ContextBuilder 子类。
    """

    def __init__(self):
        self._extra: Any = None  # prepare_generate_inputs() 的结果，子类特有输入

    def prepare_generate_inputs(self, base_inputs: GenerateInputs, **kwargs) -> None:
        """从 kwargs 中提取并校验 Pipeline 特有的输入。

        子类覆盖此方法，将结果存入 self._extra。
        默认无特有输入。
        """

    @abstractmethod
    def build_context(
        self,
        phase: InferPhase,
        inputs: GenerateInputs,
    ) -> KsanaNodeContext:
        """为指定的 InferPhase 构建 NodeContext。

        内部通过 phase.node_type 分支，为不同 Node 构建不同的 context。
        可通过 self._extra 访问 prepare_generate_inputs() 阶段提取的特有输入。
        """
        ...

    def prepare_tensors(
        self,
        phase: InferPhase,
        inputs: GenerateInputs,
    ) -> dict[TensorKey, Any] | None:
        """为指定的 InferPhase 准备需要 put 到 tensor_pool 的 tensor。

        默认返回 None。子类按需覆盖。
        """
        return None

    def check_condition(self, condition_name: str, inputs: GenerateInputs) -> bool:
        """检查条件是否满足 — 查找 self 上的同名方法。"""
        checker = getattr(self, condition_name, None)
        if checker is None:
            raise ValueError(
                f"Condition '{condition_name}' not found on {type(self).__name__}. "
                f"Add a method: def {condition_name}(self, inputs) -> bool"
            )
        return checker(inputs)

    def post_process(self, output_tensor: Any, inputs: GenerateInputs) -> Any:
        """输出后处理 — 默认直接返回。子类可覆盖。"""
        return output_tensor

    # ── 通用辅助 ──

    @staticmethod
    def _common_metadata(inputs: GenerateInputs) -> dict:
        """构建通用 metadata（offload_model, text_run_device）。"""
        return {
            "offload_model": inputs.runtime_config.offload_model,
            "text_run_device": torch.device("cpu"),
        }
