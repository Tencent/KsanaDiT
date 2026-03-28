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

"""ContextBuilder — 为 Pipeline 的每个阶段构建配置和 NodeContext。

每个 Pipeline 变体实现自己的 ContextBuilder 子类。

生命周期：

Load 阶段：
1. resolve_model_paths(model_path, ...) — 解析模型路径
2. resolve_lora_config(lora_config, ...) — 解析 LoRA 配置
3. build_loader_kwargs(model_key, ...) — 为每个 LoadTask 构建 kwargs

Generate 阶段：
1. prepare_generate_inputs(base_inputs, extra_inputs, ...) — 一次性：提取 Pipeline 特有输入
2. 对每个 NodeDef:
   a. check_condition(name, inputs) — 是否跳过
   b. build_context(node_def, inputs) — 构建 NodeContext
3. post_process(output, inputs) — 输出后处理
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kdit.config.lora_config import LoraConfig
from kdit.models.model_key import DIFFUSION_KEYS, TEXT_ENCODER_KEYS, VAE_KEYS, ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_context import NodeContext

from .extra_inputs import ExtraInputs
from .generate_inputs import PipelineGenerateInputs


class ContextBuilder(ABC):
    """为 Pipeline 的每个阶段构建配置和 NodeContext。

    每个 Pipeline 变体实现自己的 ContextBuilder 子类。
    """

    def __init__(self):
        self._extra: Any = None  # prepare_generate_inputs() 的结果，子类特有输入
        self._pipeline_def: Any = None  # Pipeline 注入的 PipelineDef 引用（用于 edges 查询）

    # ── Load 阶段 ──

    def resolve_model_paths(
        self,
        model_path: str | list[str],
        text_checkpoint_dir: str | None,
        vae_checkpoint_dir: str | None,
        pipeline_settings: Any,
    ) -> tuple[str | list[str], str, str]:
        """解析模型路径 — 默认实现处理通用逻辑。

        子类可覆盖以处理特例（如 Wan 的 high/low noise 拆分）。

        Returns:
            (load_model_path, text_checkpoint_dir, vae_checkpoint_dir)
        """
        if isinstance(model_path, (list, tuple)):
            if not Path(text_checkpoint_dir).is_dir():
                raise ValueError(
                    f"text_checkpoint_dir must be provided when loading from local checkpoint "
                    f"with diffusion model {model_path}"
                )
            if not Path(vae_checkpoint_dir).is_dir():
                raise ValueError(
                    f"vae_checkpoint_dir must be provided when loading from local checkpoint "
                    f"with diffusion model {model_path}"
                )
            return list(model_path), text_checkpoint_dir, vae_checkpoint_dir

        if Path(model_path).is_dir():
            text_checkpoint_dir = text_checkpoint_dir or model_path
            vae_checkpoint_dir = vae_checkpoint_dir or model_path
            return model_path, text_checkpoint_dir, vae_checkpoint_dir

        raise ValueError(f"model_path {model_path} should be a directory or list of diffusion model files")

    def resolve_lora_config(
        self,
        lora_config: LoraConfig | list[LoraConfig],
        pipeline_settings: Any,
    ) -> list[list[LoraConfig]]:
        """解析 LoRA 配置 — 默认实现。

        子类可覆盖以处理特例（如 Wan 的 high/low noise LoRA 拆分）。

        Returns:
            list_of_loras_list: 外层列表对应多个 diffusion checkpoint，
            内层列表对应每个 checkpoint 的多个 LoRA。
        """
        if isinstance(lora_config, LoraConfig):
            lora_list = [lora_config]
        elif isinstance(lora_config, (list, tuple)):
            lora_list = list(lora_config)
        else:
            raise ValueError(f"lora_config {lora_config} must be a LoraConfig or a list of LoraConfig")
        return [lora_list]

    def build_loader_kwargs(
        self,
        model_key: ModelKey,
        load_model_path: str | list[str],
        text_dir: str,
        vae_dir: str,
        *,
        model_config: Any,
        lora_list: list[list[LoraConfig]] | None,
        pipeline_settings: Any,
    ) -> dict:
        """构建 loader node 的 kwargs — 默认按 ModelKey 类别分发。

        子类可覆盖以处理特殊的 loader 参数。
        """
        if model_key in TEXT_ENCODER_KEYS:
            return {"model_path": text_dir}
        if model_key in DIFFUSION_KEYS:
            kwargs = {"model_path": load_model_path, "model_config": model_config}
            if lora_list:
                kwargs["lora_config"] = lora_list
            return kwargs
        if model_key in VAE_KEYS:
            vae_ckpt = getattr(pipeline_settings, "vae", None)
            vae_checkpoint = vae_ckpt.checkpoint if vae_ckpt else ""
            return {"model_path": os.path.join(vae_dir, vae_checkpoint)}
        # 未知类别 — 默认传 model_path
        return {"model_path": load_model_path}

    # ── Generate 阶段 ──

    def prepare_generate_inputs(
        self,
        base_inputs: PipelineGenerateInputs,
        extra_inputs: ExtraInputs | None,
        *,
        _default_settings: Any,
        _engine: Any,
        _vae_model_key: ModelPoolKey | None,
    ) -> None:
        """从 extra_inputs 中提取并校验 Pipeline 特有的输入。

        子类覆盖此方法，将结果存入 self._extra。
        默认无特有输入。

        Args:
            base_inputs: 公共输入（prompt, config 等）。
            extra_inputs: 模型特有输入（ExtraInputs 子类实例或 None）。
            _default_settings: Pipeline 默认配置（由 Pipeline 注入）。
            _engine: Engine 引用（由 Pipeline 注入）。
            _vae_model_key: VAE ModelPoolKey（由 Pipeline 注入）。
        """

    @abstractmethod
    def build_context(
        self,
        node_def: Any,
        inputs: PipelineGenerateInputs,
    ) -> NodeContext:
        """为指定的 NodeDef 构建 NodeContext。

        内部通过 node_def.node_type 分支，为不同 Node 构建不同的 context。
        可通过 self._extra 访问 prepare_generate_inputs() 阶段提取的特有输入。

        Args:
            node_def: NodeDef 实例（DAG 中的节点定义）。
            inputs: 公共输入。
        """
        ...

    def check_condition(self, condition_name: str, inputs: PipelineGenerateInputs) -> bool:
        """检查条件是否满足 — 查找 self 上的同名方法。"""
        checker = getattr(self, condition_name, None)
        if checker is None:
            raise ValueError(
                f"Condition '{condition_name}' not found on {type(self).__name__}. "
                f"Add a method: def {condition_name}(self, inputs) -> bool"
            )
        return checker(inputs)

    def post_process(self, output_tensor: Any, inputs: PipelineGenerateInputs) -> Any:
        """输出后处理 — 默认直接返回。子类可覆盖。"""
        return output_tensor

    # ── 通用辅助 ──

    @staticmethod
    def _common_metadata(inputs: PipelineGenerateInputs) -> dict:
        """构建通用 metadata（offload_model）。"""
        return {
            "offload_model": inputs.runtime_config.offload_model,
        }
