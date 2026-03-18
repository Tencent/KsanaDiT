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

"""Wan 系列 ContextBuilder — 为 Wan T2V / I2V / VACE 构建 NodeContext。

WanContextBuilder 是公共基类，提供 Wan 系列共用的 context 构建方法。
每个变体（T2V、I2V、VACE）实现自己的子类。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as tvtf
from PIL import Image

from kdit.config.lora_config import LoraConfig
from kdit.engine import Engine
from kdit.models.model_key import ModelKey
from kdit.models.vae_model import compute_video_latent_shape
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.tensor import TensorKey
from kdit.utils.logger import log
from kdit.utils.types import str_to_list
from kdit.utils.vace import VaceConfig, build_vace_video_control_config, latent_process_out

from ..context_builder import ContextBuilder
from ..generate_inputs import PipelineGenerateInputs
from ..pipeline_def import InferPhase

# ── 公共基类 ─────────────────────────────────────────────────────────────


class WanContextBuilder(ContextBuilder):
    """Wan 系列的公共基类 — 提供共用的 context 构建方法。

    子类只需实现 ``prepare_generate_inputs`` 和 ``build_context``。

    覆盖 Load 阶段方法以处理 Wan 特有的 high/low noise 拆分逻辑。
    """

    # ── Load 阶段覆盖 ──

    def resolve_model_paths(
        self,
        model_path: str | list[str],
        text_checkpoint_dir: str | None,
        vae_checkpoint_dir: str | None,
        pipeline_settings: Any,
    ) -> tuple[str | list[str], str, str]:
        """Wan 系列：目录模式下自动拆分 high/low noise checkpoint。

        通过 settings 中是否有 ``high_noise_checkpoint`` / ``low_noise_checkpoint``
        来判断是否需要拆分，不再硬编码 PipelineKey。
        """
        load_model_path, text_dir, vae_dir = super().resolve_model_paths(
            model_path, text_checkpoint_dir, vae_checkpoint_dir, pipeline_settings
        )

        # Wan 特例：如果 settings 中有 high/low noise checkpoint 配置，自动拆分
        diffusion = getattr(pipeline_settings, "diffusion", None)
        high = getattr(diffusion, "high_noise_checkpoint", None) if diffusion else None
        low = getattr(diffusion, "low_noise_checkpoint", None) if diffusion else None
        if high and low and isinstance(load_model_path, str):
            load_model_path = [
                os.path.join(load_model_path, high),
                os.path.join(load_model_path, low),
            ]

        return load_model_path, text_dir, vae_dir

    def resolve_lora_config(
        self,
        lora_config: LoraConfig | list[LoraConfig],
        pipeline_settings: Any,
    ) -> list[list[LoraConfig]]:
        """Wan 系列：自动拆分 high/low noise LoRA。

        通过 settings 中是否有 ``high_noise_lora_checkpoint`` / ``low_noise_lora_checkpoint``
        来判断是否需要拆分，不再硬编码 PipelineKey。
        """
        if isinstance(lora_config, LoraConfig):
            lora_list = [lora_config]
        elif isinstance(lora_config, (list, tuple)):
            lora_list = list(lora_config)
        else:
            raise ValueError(f"lora_config {lora_config} must be a LoraConfig or a list of LoraConfig")

        diffusion = getattr(pipeline_settings, "diffusion", None)
        high_lora_ckpt = getattr(diffusion, "high_noise_lora_checkpoint", None) if diffusion else None
        low_lora_ckpt = getattr(diffusion, "low_noise_lora_checkpoint", None) if diffusion else None

        if high_lora_ckpt and low_lora_ckpt:
            # 拆分为 high/low noise LoRA 列表
            lora_list_high = []
            lora_list_low = []
            for one_lora in lora_list:
                if not isinstance(one_lora, LoraConfig):
                    raise ValueError(f"one_lora {one_lora} must be a LoraConfig")
                if not Path(one_lora.path).is_dir():
                    raise ValueError(f"one_lora.path {one_lora.path} must be a directory for high/low noise LoRA")
                lora_list_high.append(
                    LoraConfig(
                        path=os.path.join(one_lora.path, high_lora_ckpt),
                        strength=one_lora.strength,
                    )
                )
                lora_list_low.append(
                    LoraConfig(
                        path=os.path.join(one_lora.path, low_lora_ckpt),
                        strength=one_lora.strength,
                    )
                )
            return [lora_list_high, lora_list_low]

        return [lora_list]

    # ── Generate 阶段 ──

    def _build_text_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 TextEncode 的 context。"""
        return NodeContext(
            prompt=inputs.prompt,
            negative_prompt=inputs.prompt_negative,
            metadata=self._common_metadata(inputs),
        )

    def _build_gen_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 Generator 的 context。"""
        extra = self._extra
        return NodeContext(
            sample_config=inputs.sample_config,
            runtime_config=inputs.runtime_config,
            cache_config=inputs.cache_config,
            metadata={
                "noise_shape": getattr(extra, "noise_shape", None),
                "control_video_config": getattr(extra, "vace_video_control_config", None),
            },
        )

    def _build_decode_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 VAE Decode 的 context。"""
        extra = self._extra
        return NodeContext(
            metadata={
                "offload_model": inputs.runtime_config.offload_model,
                "with_end_image": getattr(extra, "with_end_image", False),
            },
        )

    def _build_save_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 SaveNode 的 context — 包含保存路径和 fps。"""
        from . import compute_save_path

        return NodeContext(
            metadata={
                "save_path": compute_save_path(inputs, prefix="wan", ext=".mp4"),
                "fps": getattr(self._extra, "fps", 16),
            },
        )


# ── T2V ──────────────────────────────────────────────────────────────────


class WanT2VContextBuilder(WanContextBuilder):
    """Wan T2V — 纯文生视频。

    prepare_generate_inputs 计算 noise_shape。
    build_context 按 node_type 分发到基类的 _build_*_ctx 方法。
    """

    @dataclass
    class ExtraPipelineGenerateInputs:
        """T2V 特有的中间数据。"""

        noise_shape: list[int]
        fps: int = 16

    def prepare_generate_inputs(self, base_inputs: PipelineGenerateInputs, **kwargs) -> None:
        """计算 noise_shape 并存入 _extra。

        需要 kwargs 中的 ``_default_settings`` 来获取 VAE 参数。
        """
        settings = kwargs.get("_default_settings")
        if settings is None:
            raise ValueError(
                "WanT2VContextBuilder requires '_default_settings' in kwargs. "
                "This should be injected by Pipeline.generate()."
            )

        rc = base_inputs.runtime_config
        noise_shape = list(
            compute_video_latent_shape(
                z_dim=settings.vae.z_dim,
                target_f=rc.frame_num,
                target_h=rc.size[1],
                target_w=rc.size[0],
                vae_stride=list(settings.vae.stride),
                vae_patch=list(settings.diffusion.patch_size),
            )
        )
        fps = getattr(settings.vae, "fps", 16)
        self._extra = self.ExtraPipelineGenerateInputs(noise_shape=noise_shape, fps=fps)

    def build_context(self, phase: InferPhase, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 node_type 分发构建 context。"""
        match phase.node_type:
            case NT.TEXT_ENCODE:
                return self._build_text_ctx(inputs)
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_VIDEO:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"WanT2VContextBuilder: unexpected node_type {phase.node_type}")


# ── I2V ──────────────────────────────────────────────────────────────────


class WanI2VContextBuilder(WanContextBuilder):
    """Wan I2V — 图生视频（含可选 VACE 控制）。

    处理 start_img_path / end_img_path / input_latent / video_control_config。
    当有 start_img 时走 VAE_ENCODE_SPATIAL 编码图像；无图时退化为 T2V 行为。
    """

    @dataclass
    class ExtraPipelineGenerateInputs:
        """I2V 特有的中间数据。"""

        start_img_path: list[str] | None
        end_img_path: list[str] | None
        start_img_tensor: torch.Tensor | None
        end_img_tensor: torch.Tensor | None
        input_latent: torch.Tensor | None
        target_frame_num: int
        noise_shape: list[int] | None
        with_end_image: bool
        vace_video_control_config: VaceConfig | None
        fps: int = 16

    def prepare_generate_inputs(self, base_inputs: PipelineGenerateInputs, **kwargs) -> None:
        """提取 I2V 特有输入：图片路径、VACE 配置、noise_shape。

        需要 kwargs 中的:
        - ``_default_settings``: VAE 参数
        - ``_engine``: Engine 实例（VACE 的 vae_encode_fn 需要）
        - ``_vae_model_key``: VAE 模型 key（VACE 的 vae_encode_fn 需要）
        """
        settings = kwargs.get("_default_settings")
        if settings is None:
            raise ValueError(
                "WanI2VContextBuilder requires '_default_settings' in kwargs. "
                "This should be injected by Pipeline.generate()."
            )

        rc = base_inputs.runtime_config
        num_prompts = base_inputs.num_prompts

        # 校验图片路径
        start_img_path = _valid_images(kwargs.get("start_img_path"), num_prompts)
        end_img_path = _valid_images(kwargs.get("end_img_path"), num_prompts)
        with_end_image = end_img_path is not None

        # VACE 配置
        vace_video_control_config = _valid_video_control_config(
            video_control_config=kwargs.get("video_control_config"),
            runtime_config=rc,
            engine=kwargs.get("_engine"),
            vae_model_key=kwargs.get("_vae_model_key"),
            vae_stride=getattr(settings.vae, "stride", None),
        )

        # 计算 target_frame_num（VACE 可能调整帧数）
        target_frame_num = (
            vace_video_control_config.adjusted_frame_num
            if vace_video_control_config and vace_video_control_config.adjusted_frame_num
            else rc.frame_num
        )

        # noise_shape：有图时为 None（由 GeneratorNode 从 image_embeds 推导）
        noise_shape = (
            None
            if start_img_path is not None
            else list(
                compute_video_latent_shape(
                    z_dim=settings.vae.z_dim,
                    target_f=target_frame_num,
                    target_h=rc.size[1],
                    target_w=rc.size[0],
                    vae_stride=list(settings.vae.stride),
                    vae_patch=list(settings.diffusion.patch_size),
                )
            )
        )

        # 预加载图片 tensor
        start_img_tensor, end_img_tensor = None, None
        if start_img_path is not None:
            start_img_tensor = _load_image(start_img_path)
            end_img_tensor = _load_image(end_img_path) if end_img_path is not None else None

        fps = getattr(settings.vae, "fps", 16)
        self._extra = self.ExtraPipelineGenerateInputs(
            start_img_path=start_img_path,
            end_img_path=end_img_path,
            start_img_tensor=start_img_tensor,
            end_img_tensor=end_img_tensor,
            input_latent=kwargs.get("input_latent"),
            target_frame_num=target_frame_num,
            noise_shape=noise_shape,
            with_end_image=with_end_image,
            vace_video_control_config=vace_video_control_config,
            fps=fps,
        )

    def build_context(self, phase: InferPhase, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 node_type 分发构建 context。"""
        extra = self._extra
        match phase.node_type:
            case NT.TEXT_ENCODE:
                return self._build_text_ctx(inputs)
            case NT.VAE_ENCODE_SPATIAL:
                return NodeContext(
                    metadata={
                        "target_f": extra.target_frame_num,
                        "target_h": inputs.runtime_config.size[1],
                        "target_w": inputs.runtime_config.size[0],
                    }
                )
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_VIDEO:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"WanI2VContextBuilder: unexpected node_type {phase.node_type}")

    def prepare_tensors(self, phase: InferPhase, inputs: PipelineGenerateInputs) -> dict[TensorKey, Any] | None:
        """为 VAE_ENCODE_SPATIAL 和 GENERATE 阶段准备 tensor。"""
        extra = self._extra
        if phase.node_type == NT.VAE_ENCODE_SPATIAL:
            return {
                TensorKey.START_IMG: extra.start_img_tensor,
                TensorKey.END_IMG: extra.end_img_tensor,
            }
        if phase.node_type == NT.GENERATE and extra.input_latent is not None:
            return {TensorKey.INPUT_LATENT: extra.input_latent}
        return None

    def has_start_image(self, inputs: PipelineGenerateInputs) -> bool:
        """条件方法：是否有起始图 — 用于 PipelineDef 的 .when("has_start_image")。"""
        return self._extra.start_img_path is not None


# ── 辅助函数 ─────────────────────────────────────────────────────────────


def _valid_images(img_path: str | list[str] | None, num_prompts: int) -> list[str] | None:
    """校验图片路径列表 — 从 BasePipeline._valid_images 迁移。"""
    if img_path is None:
        return None
    img_path = str_to_list(img_path)
    if len(img_path) != 1 and len(img_path) != num_prompts:
        raise ValueError(
            f"img_path length ({len(img_path)}) must match prompt list length ({num_prompts}) or only one image"
        )
    return img_path


def _load_image(img_paths: list[str] | None, device: str = "cpu") -> torch.Tensor | None:
    """加载图片列表为 tensor — 从 BasePipeline._load_image 迁移。

    Returns:
        [B, C, H, W] tensor，归一化到 [-1, 1]。
    """
    if img_paths is None:
        return None
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


def _valid_video_control_config(
    video_control_config: VaceConfig | None,
    runtime_config: Any,
    engine: Engine | None,
    vae_model_key: ModelKey | None,
    vae_stride: Any | None,
) -> VaceConfig | None:
    """校验并构建 VACE 配置 — 从 BasePipeline._valid_video_control_config 迁移。

    需要 engine 和 vae_model_key 来构建 vae_encode_fn 闭包。
    """
    if video_control_config is None:
        return None
    if engine is None or vae_model_key is None:
        raise ValueError(
            "WanI2VContextBuilder requires '_engine' and '_vae_model_key' in kwargs "
            "for VACE video_control_config processing."
        )

    width, height = runtime_config.size
    num_frames = runtime_config.frame_num

    def vae_encode_fn(frame: torch.Tensor) -> torch.Tensor:
        context = NodeContext()
        with engine.tensor_scope(keep=[TensorKey.IMAGE_EMBEDS]):
            engine.put_tensors(**{TensorKey.IMAGE: frame})
            engine.run_infer_node(NT.VAE_ENCODE_IMAGES, vae_model_key, context)
        tensor_value = engine.get_tensor(TensorKey.IMAGE_EMBEDS)
        latents_list = tensor_value.data  # list[Tensor]
        return latent_process_out(latents_list[0])

    vace_config = build_vace_video_control_config(
        video_control_config=video_control_config,
        width=width,
        height=height,
        num_frames=num_frames,
        vae_encode_fn=vae_encode_fn,
    )

    if vace_config is not None and vace_config.trim_latent > 0:
        vae_temporal_stride = vae_stride[0] if vae_stride is not None else 4
        adjusted_frame_num = num_frames + vace_config.trim_latent * vae_temporal_stride
        vace_config.adjusted_frame_num = adjusted_frame_num
    elif vace_config is not None:
        vace_config.adjusted_frame_num = num_frames

    return vace_config
