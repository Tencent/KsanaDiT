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

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torchvision.transforms.functional as tvtf
from PIL import Image

from kdit.models.vae_model import compute_image_latent_shape
from kdit.nodes.core.node_context import KsanaNodeContext
from kdit.nodes.core.node_types import KsanaInferNodeType as NT
from kdit.tensor import TensorKey
from kdit.utils.logger import log

from ..context_builder import ContextBuilder
from ..generate_inputs import GenerateInputs
from ..pipeline_def import InferPhase

# ── 公共基类 ─────────────────────────────────────────────────────────────


class QwenContextBuilder(ContextBuilder):
    """Qwen 系列的公共基类 — 提供共用的 context 构建方法。

    子类只需实现 ``prepare_generate_inputs`` 和 ``build_context``。
    """

    def _build_text_ctx(self, inputs: GenerateInputs, condition_image_path: Any = None) -> KsanaNodeContext:
        """构建 TextEncode 的 context。

        Args:
            condition_image_path: Edit 模式下的条件图路径（传入 metadata）。
        """
        metadata = self._common_metadata(inputs)
        if condition_image_path is not None:
            metadata["condition_image_path"] = condition_image_path
        return KsanaNodeContext(
            prompt=inputs.prompt,
            negative_prompt=inputs.prompt_negative,
            metadata=metadata,
        )

    def _build_gen_ctx(self, inputs: GenerateInputs) -> KsanaNodeContext:
        """构建 Generator 的 context。"""
        extra = self._extra
        return KsanaNodeContext(
            sample_config=inputs.sample_config,
            runtime_config=inputs.runtime_config,
            cache_config=inputs.cache_config,
            metadata={
                "noise_shape": getattr(extra, "noise_shape", None),
            },
        )

    def _build_decode_ctx(self, inputs: GenerateInputs) -> KsanaNodeContext:
        """构建 VAE Decode 的 context。"""
        return KsanaNodeContext(
            metadata={
                "offload_model": inputs.runtime_config.offload_model,
            },
        )

    def _build_save_ctx(self, inputs: GenerateInputs) -> KsanaNodeContext:
        """构建 SaveImage 的 context — 包含保存路径。"""
        return KsanaNodeContext(
            metadata={
                "save_path": _compute_save_path(inputs),
            },
        )


# ── T2I ──────────────────────────────────────────────────────────────────


class QwenT2IContextBuilder(QwenContextBuilder):
    """Qwen T2I — 纯文生图。

    prepare_generate_inputs 计算 noise_shape。
    build_context 按 node_type 分发到基类的 _build_*_ctx 方法。
    """

    @dataclass
    class Extra:
        """T2I 特有的中间数据。"""

        noise_shape: list[int]

    def prepare_generate_inputs(self, base_inputs: GenerateInputs, **kwargs) -> None:
        """计算 noise_shape 并存入 _extra。"""
        settings = kwargs.get("_default_settings")
        if settings is None:
            raise ValueError(
                "QwenT2IContextBuilder requires '_default_settings' in kwargs. "
                "This should be injected by Pipeline.generate()."
            )

        rc = base_inputs.runtime_config
        noise_shape = list(
            compute_image_latent_shape(
                z_dim=settings.vae.z_dim,
                target_h=rc.size[1],
                target_w=rc.size[0],
                vae_scale_factor=settings.vae.vae_scale_factor,
                patch_size=settings.diffusion.patch_size,
            )
        )
        self._extra = self.Extra(noise_shape=noise_shape)

    def build_context(self, phase: InferPhase, inputs: GenerateInputs) -> KsanaNodeContext:
        """按 node_type 分发构建 context。"""
        match phase.node_type:
            case NT.TEXT_ENCODE:
                return self._build_text_ctx(inputs)
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_IMAGE:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"QwenT2IContextBuilder: unexpected node_type {phase.node_type}")


# ── Edit ─────────────────────────────────────────────────────────────────


class QwenEditContextBuilder(QwenContextBuilder):
    """Qwen Edit — 图像编辑（参考图 + 文本指令）。

    处理 img_path（参考图路径列表）和 input_latent。
    当有 img_path 时走 VAE_ENCODE_IMAGES 编码参考图。
    """

    @dataclass
    class Extra:
        """Edit 特有的中间数据。"""

        img_path: list[list[str]] | None
        img_tensor: list[torch.Tensor] | torch.Tensor | None
        input_latent: torch.Tensor | None
        noise_shape: list[int]

    def prepare_generate_inputs(self, base_inputs: GenerateInputs, **kwargs) -> None:
        """提取 Edit 特有输入：参考图路径、noise_shape。"""
        settings = kwargs.get("_default_settings")
        if settings is None:
            raise ValueError(
                "QwenEditContextBuilder requires '_default_settings' in kwargs. "
                "This should be injected by Pipeline.generate()."
            )

        rc = base_inputs.runtime_config
        num_prompts = base_inputs.num_prompts

        # 校验参考图路径
        img_path = _valid_ref_images(kwargs.get("img_path"), num_prompts)

        # 计算 noise_shape（Qwen 始终显式计算）
        noise_shape = list(
            compute_image_latent_shape(
                z_dim=settings.vae.z_dim,
                target_h=rc.size[1],
                target_w=rc.size[0],
                vae_scale_factor=settings.vae.vae_scale_factor,
                patch_size=settings.diffusion.patch_size,
            )
        )

        # 预加载参考图 tensor
        img_tensor = None
        if img_path is not None:
            img_tensor = _load_ref_images(img_path)

        self._extra = self.Extra(
            img_path=img_path,
            img_tensor=img_tensor,
            input_latent=kwargs.get("input_latent"),
            noise_shape=noise_shape,
        )

    def build_context(self, phase: InferPhase, inputs: GenerateInputs) -> KsanaNodeContext:
        """按 node_type 分发构建 context。"""
        extra = self._extra
        match phase.node_type:
            case NT.TEXT_ENCODE:
                # Edit 模式：传入 condition_image_path
                return self._build_text_ctx(inputs, condition_image_path=extra.img_path)
            case NT.VAE_ENCODE_IMAGES:
                return KsanaNodeContext()
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_IMAGE:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"QwenEditContextBuilder: unexpected node_type {phase.node_type}")

    def prepare_tensors(self, phase: InferPhase, inputs: GenerateInputs) -> dict[TensorKey, Any] | None:
        """为 VAE_ENCODE_IMAGES 和 GENERATE 阶段准备 tensor。"""
        extra = self._extra
        if phase.node_type == NT.VAE_ENCODE_IMAGES and extra.img_tensor is not None:
            return {TensorKey.IMAGE: extra.img_tensor}
        if phase.node_type == NT.GENERATE and extra.input_latent is not None:
            return {TensorKey.INPUT_LATENT: extra.input_latent}
        return None

    def has_ref_images(self, inputs: GenerateInputs) -> bool:
        """条件方法：是否有参考图 — 用于 PipelineDef 的 .when("has_ref_images")。"""
        return self._extra.img_path is not None


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def _compute_save_path(inputs: GenerateInputs) -> str | None:
    """从 runtime_config 计算保存路径。

    如果 save_output=False，返回 None（SaveNode 会跳过保存）。
    """
    rc = inputs.runtime_config
    if not rc.save_output:
        return None

    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_text = inputs.prompt if isinstance(inputs.prompt, str) else inputs.prompt[0]
    formatted_prompt = prompt_text.replace(" ", "_").replace("/", "_")[:30]
    out_size = rc.size
    filename = f"qwen_w{out_size[0]}_h{out_size[1]}_{formatted_time}_{formatted_prompt}_0.png"
    return os.path.join(rc.output_folder, filename)


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


def _load_image(img_paths: list[str], device: str = "cpu") -> torch.Tensor:
    """加载图片列表为 tensor — [B, C, H, W]，归一化到 [-1, 1]。"""
    log.info(f"load input image: {img_paths}")
    imgs = []
    shape = None
    for one_path in img_paths:
        img = Image.open(one_path).convert("RGB")
        if shape is None:
            shape = img.size
        elif img.size != shape:
            raise ValueError(f"all images {img_paths} should have the same shape, but got {img.size} and {shape}")
        img = tvtf.to_tensor(img).sub_(0.5).div_(0.5).to(device)
        imgs.append(img.unsqueeze(0))
    if len(imgs) == 1:
        return imgs[0]
    return torch.cat(imgs, dim=0)


def _load_ref_images(
    img_path: list[list[str]],
    device: str = "cpu",
) -> list[torch.Tensor] | torch.Tensor:
    """加载参考图 — 二维列表返回 list[Tensor]，每个 Tensor 是一组参考图。"""
    return [_load_image(paths, device=device) for paths in img_path]
