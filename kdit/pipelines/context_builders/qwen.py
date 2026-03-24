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

"""Qwen 系列 ContextBuilder — 为 Qwen T2I / Edit 构建 NodeContext。

QwenContextBuilder 是公共基类，提供 Qwen 系列共用的 context 构建方法。
QwenT2IContextBuilder 和 QwenEditContextBuilder 分别处理纯文生图和编辑模式。
"""

from dataclasses import dataclass
from typing import Any

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT

from ..context_builder import ContextBuilder
from ..extra_inputs import ExtraInputs
from ..generate_inputs import PipelineGenerateInputs
from ..pipeline_def import NodeDef

# ── ExtraInputs 子类 ─────────────────────────────────────────────────────


@dataclass
class QwenEditExtraInputs(ExtraInputs):
    """Qwen Edit 的模型特有输入。

    用法::

        pipeline.generate(
            prompt,
            extra_inputs=QwenEditExtraInputs(img_path=["ref.jpg"]),
        )
    """

    img_path: str | list[str] | list[list[str]] | None = None


# ── 公共基类 ─────────────────────────────────────────────────────────────


class QwenContextBuilder(ContextBuilder):
    """Qwen 系列的公共基类 — 提供共用的 context 构建方法。

    子类只需实现 ``prepare_generate_inputs`` 和 ``build_context``。
    """

    def _build_text_ctx(self, inputs: PipelineGenerateInputs, condition_image_path: Any = None) -> NodeContext:
        """构建 TextEncode 的 context。

        Args:
            condition_image_path: Edit 模式下的条件图路径（传入 metadata）。
        """
        metadata = self._common_metadata(inputs)
        if condition_image_path is not None:
            metadata["condition_image_path"] = condition_image_path
        return NodeContext(
            prompt=inputs.prompt,
            negative_prompt=inputs.prompt_negative,
            metadata=metadata,
        )

    def _build_gen_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 Generator 的 context。"""
        return NodeContext(
            sample_config=inputs.sample_config,
            runtime_config=inputs.runtime_config,
            cache_config=inputs.cache_config,
            metadata={},
        )

    def _build_decode_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 VAE Decode 的 context。"""
        return NodeContext(
            metadata={
                "offload_model": inputs.runtime_config.offload_model,
            },
        )

    def _build_save_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 SaveImage 的 context — 包含保存路径。"""
        from . import compute_save_path

        return NodeContext(
            metadata={
                "save_path": compute_save_path(inputs, prefix="qwen", ext=".png"),
            },
        )


# ── T2I ──────────────────────────────────────────────────────────────────


class QwenT2IContextBuilder(QwenContextBuilder):
    """Qwen T2I — 纯文生图。

    prepare_generate_inputs 计算 noise_shape。
    build_context 按 node_type 分发到基类的 _build_*_ctx 方法。
    """

    @dataclass
    class _Extra:
        """T2I 特有的中间数据。"""

        target_h: int
        target_w: int

    def prepare_generate_inputs(
        self,
        base_inputs: PipelineGenerateInputs,
        extra_inputs: ExtraInputs | None,
        *,
        _default_settings: Any,
        _engine: Any,
        _vae_model_key: ModelKey | None,
    ) -> None:
        """保存目标尺寸，noise_shape 由 VAE_COMPUTE_SHAPE 节点计算。"""
        rc = base_inputs.runtime_config
        self._extra = self._Extra(
            target_h=rc.size[1],
            target_w=rc.size[0],
        )

    def build_context(self, node_def: NodeDef, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 node_type 分发构建 context。"""
        extra = self._extra
        match node_def.node_type:
            case NT.TEXT_ENCODE:
                return self._build_text_ctx(inputs)
            case NT.VAE_COMPUTE_SHAPE:
                return NodeContext(
                    metadata={
                        "target_f": 1,
                        "target_h": extra.target_h,
                        "target_w": extra.target_w,
                    }
                )
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_IMAGE:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"QwenT2IContextBuilder: unexpected node_type {node_def.node_type}")


# ── Edit ─────────────────────────────────────────────────────────────────


class QwenEditContextBuilder(QwenContextBuilder):
    """Qwen Edit — 图像编辑（参考图 + 文本指令）。

    处理 img_path（参考图路径列表）。
    当有 img_path 时走 ReadImageNode → VAE_ENCODE_IMAGES 编码参考图。
    图片加载由 ReadImageNode 在 DAG 中完成，不再在 prepare_generate_inputs 中预加载。
    """

    @dataclass
    class _Extra:
        """Edit 特有的中间数据（从 ExtraInputs + settings 计算得出）。"""

        img_path: list[list[str]] | None
        target_h: int
        target_w: int

    def prepare_generate_inputs(
        self,
        base_inputs: PipelineGenerateInputs,
        extra_inputs: ExtraInputs | None,
        *,
        _default_settings: Any,
        _engine: Any,
        _vae_model_key: ModelKey | None,
    ) -> None:
        """提取 Edit 特有输入：参考图路径、目标尺寸。"""
        rc = base_inputs.runtime_config
        num_prompts = base_inputs.num_prompts

        # 从 ExtraInputs 提取
        ei = extra_inputs if isinstance(extra_inputs, QwenEditExtraInputs) else QwenEditExtraInputs()

        # 校验参考图路径
        img_path = _valid_ref_images(ei.img_path, num_prompts)

        self._extra = self._Extra(
            img_path=img_path,
            target_h=rc.size[1],
            target_w=rc.size[0],
        )

    def build_context(self, node_def: NodeDef, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 node_type 分发构建 context。"""
        extra = self._extra
        match node_def.node_type:
            case NT.TEXT_ENCODE:
                # Edit 模式：传入 condition_image_path
                return self._build_text_ctx(inputs, condition_image_path=extra.img_path)
            case NT.READ_IMAGE:
                # ReadImageNode: 传入参考图路径
                return NodeContext(metadata={"img_paths": extra.img_path})
            case NT.VAE_COMPUTE_SHAPE:
                return NodeContext(
                    metadata={
                        "target_f": 1,
                        "target_h": extra.target_h,
                        "target_w": extra.target_w,
                    }
                )
            case NT.VAE_ENCODE_IMAGES:
                return NodeContext()
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_IMAGE:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"QwenEditContextBuilder: unexpected node_type {node_def.node_type}")

    def has_ref_images(self, inputs: PipelineGenerateInputs) -> bool:
        """条件方法：是否有参考图 — 用于 PipelineDef 的 .when("has_ref_images")。"""
        return self._extra.img_path is not None


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def _valid_ref_images(
    img_path: str | list[str] | list[list[str]] | None,
    num_prompts: int,
) -> list[list[str]] | None:
    """校验参考图路径（支持二维列表）— 从 BasePipeline._valid_ref_images 迁移。"""
    if img_path is None:
        return None

    # 统一为二维列表
    if isinstance(img_path, str):
        img_path = [[img_path]]
    elif isinstance(img_path, list) and img_path and isinstance(img_path[0], str):
        img_path = [img_path]

    if len(img_path) != 1 and len(img_path) != num_prompts:
        raise ValueError(
            f"img_path length ({len(img_path)}) must match prompt list length ({num_prompts}) or only one group"
        )
    return img_path
