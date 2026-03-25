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
from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.tensor import TensorKey
from kdit.utils.logger import log
from kdit.utils.types import str_to_list
from kdit.utils.vace import VaceConfig, build_vace_video_control_config, latent_process_out

from ..context_builder import ContextBuilder
from ..extra_inputs import ExtraInputs
from ..generate_inputs import PipelineGenerateInputs
from ..pipeline_def import NodeDef

# ── ExtraInputs 子类 ─────────────────────────────────────────────────────


@dataclass
class WanI2VExtraInputs(ExtraInputs):
    """Wan I2V / VACE 的模型特有输入。

    用法::

        pipeline.generate(
            prompt,
            extra_inputs=WanI2VExtraInputs(start_img_path="a.jpg"),
        )
    """

    start_img_path: str | list[str] | None = None
    end_img_path: str | list[str] | None = None
    video_control_config: VaceConfig | None = None
    aux_latent: torch.Tensor | None = None


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
        metadata = self._common_metadata(inputs)
        metadata["text_run_device"] = torch.device("cpu")
        return NodeContext(
            prompt=inputs.prompt,
            negative_prompt=inputs.prompt_negative,
            metadata=metadata,
        )

    def _build_gen_ctx(self, inputs: PipelineGenerateInputs) -> NodeContext:
        """构建 Generator 的 context。"""
        extra = self._extra
        return NodeContext(
            sample_config=inputs.sample_config,
            runtime_config=inputs.runtime_config,
            cache_config=inputs.cache_config,
            metadata={
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
    class _Extra:
        """T2V 特有的中间数据。"""

        target_f: int
        target_h: int
        target_w: int
        fps: int = 16

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
        fps = getattr(_default_settings.vae, "fps", 16)
        self._extra = self._Extra(
            target_f=rc.frame_num,
            target_h=rc.size[1],
            target_w=rc.size[0],
            fps=fps,
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
                        "target_f": extra.target_f,
                        "target_h": extra.target_h,
                        "target_w": extra.target_w,
                    }
                )
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_VIDEO:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"WanT2VContextBuilder: unexpected node_type {node_def.node_type}")


# ── I2V ──────────────────────────────────────────────────────────────────


class WanI2VContextBuilder(WanContextBuilder):
    """Wan I2V — 图生视频（含可选 VACE 控制）。

    处理 start_img_path / end_img_path / aux_latent / video_control_config。
    当有 start_img 时走 VAE_ENCODE_SPATIAL 编码图像；无图时退化为 T2V 行为。
    """

    @dataclass
    class _Extra:
        """I2V 特有的中间数据（从 ExtraInputs + settings 计算得出）。"""

        start_img_path: list[str] | None
        end_img_path: list[str] | None
        target_frame_num: int
        with_end_image: bool
        vace_video_control_config: VaceConfig | None
        fps: int = 16

    def prepare_generate_inputs(
        self,
        base_inputs: PipelineGenerateInputs,
        extra_inputs: ExtraInputs | None,
        *,
        _default_settings: Any,
        _engine: Any,
        _vae_model_key: ModelKey | None,
    ) -> None:
        """提取 I2V 特有输入：图片路径、VACE 配置、noise_shape。"""
        settings = _default_settings
        rc = base_inputs.runtime_config
        num_prompts = base_inputs.num_prompts

        # 从 ExtraInputs 提取
        ei = extra_inputs if isinstance(extra_inputs, WanI2VExtraInputs) else WanI2VExtraInputs()

        # 校验图片路径
        start_img_path = _valid_images(ei.start_img_path, num_prompts)
        end_img_path = _valid_images(ei.end_img_path, num_prompts)
        with_end_image = end_img_path is not None

        # VACE 配置
        vace_video_control_config = _valid_video_control_config(
            video_control_config=ei.video_control_config,
            runtime_config=rc,
            engine=_engine,
            vae_model_key=_vae_model_key,
            vae_stride=getattr(settings.vae, "stride", None),
        )

        # 计算 target_frame_num（VACE 可能调整帧数）
        target_frame_num = (
            vace_video_control_config.adjusted_frame_num
            if vace_video_control_config and vace_video_control_config.adjusted_frame_num
            else rc.frame_num
        )

        fps = getattr(settings.vae, "fps", 16)
        self._extra = self._Extra(
            start_img_path=start_img_path,
            end_img_path=end_img_path,
            target_frame_num=target_frame_num,
            with_end_image=with_end_image,
            vace_video_control_config=vace_video_control_config,
            fps=fps,
        )

    def build_context(self, node_def: NodeDef, inputs: PipelineGenerateInputs) -> NodeContext:
        """按 node_type 分发构建 context。"""
        extra = self._extra
        match node_def.node_type:
            case NT.TEXT_ENCODE:
                return self._build_text_ctx(inputs)
            case NT.READ_IMAGE:
                # ReadImageNode: 通过 edges 区分 start/end image
                img_paths = self._resolve_read_image_paths(node_def)
                return NodeContext(metadata={"img_paths": img_paths})
            case NT.VAE_ENCODE_SPATIAL:
                return NodeContext(
                    metadata={
                        "target_f": extra.target_frame_num,
                        "target_h": inputs.runtime_config.size[1],
                        "target_w": inputs.runtime_config.size[0],
                    }
                )
            case NT.VACE_PREPROCESS:
                return NodeContext(
                    metadata={"vace_config": extra.vace_video_control_config},
                )
            case NT.GENERATE:
                return self._build_gen_ctx(inputs)
            case NT.VAE_DECODE:
                return self._build_decode_ctx(inputs)
            case NT.SAVE_VIDEO:
                return self._build_save_ctx(inputs)
            case _:
                raise ValueError(f"WanI2VContextBuilder: unexpected node_type {node_def.node_type}")

    def _resolve_read_image_paths(self, node_def: NodeDef) -> list[str] | None:
        """通过 DAG edges 区分多个 ReadImageNode 实例。

        查看 node_def 的出边连接到哪个 dst_pin：
        - 连接到 START_IMG → 返回 start_img_path
        - 连接到 END_IMG → 返回 end_img_path
        """
        extra = self._extra
        if self._pipeline_def is None:
            # 无 pipeline_def 时回退到 start_img_path
            return extra.start_img_path

        for edge in self._pipeline_def.edges:
            if edge.src_node_id == node_def.node_id and isinstance(edge.src_pin, TensorKey):
                if edge.dst_pin == TensorKey.START_IMG:
                    return extra.start_img_path
                if edge.dst_pin == TensorKey.END_IMG:
                    return extra.end_img_path
        return extra.start_img_path

    def has_start_image(self, inputs: PipelineGenerateInputs) -> bool:
        """条件方法：是否有起始图 — 用于 PipelineDef 的 .when("has_start_image")。"""
        return self._extra.start_img_path is not None

    def has_vace(self, inputs: PipelineGenerateInputs) -> bool:
        """条件方法：是否有 VACE 配置 — 用于 PipelineDef 的 .when("has_vace")。"""
        return self._extra.vace_video_control_config is not None


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
    engine: Any | None,
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
        try:
            engine.put_tensors({TensorKey.IMAGE: frame})
            engine.run_infer_node(NT.VAE_ENCODE_IMAGES, vae_model_key, context)
            tensor_value = engine.get_tensor(TensorKey.AUX_LATENT)
        finally:
            engine.clear_all_tensors()
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
