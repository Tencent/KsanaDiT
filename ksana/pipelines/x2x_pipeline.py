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

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from ..config import KsanaDistributedConfig, KsanaModelConfig, KsanaRuntimeConfig, KsanaSampleConfig
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
from ..config.lora_config import KsanaLoraConfig
from ..engine import get_engine
from ..models.model_base import ModelBase
from ..models.model_key import KsanaModelKey, get_model_key_from_path
from ..models.vae_model import compute_image_latent_shape, compute_video_latent_shape
from ..nodes.core.node_context import KsanaNodeContext
from ..nodes.core.node_types import KsanaInferNodeType
from ..settings import load_default_settings
from ..tensor import KsanaTensorKey
from ..utils import log, time_range
from ..utils.media import save_image
from ..utils.monitor import report
from ..utils.vace import KsanaVaceContext
from .base_pipeline import KsanaBasePipeline


class KsanaPipeline(KsanaBasePipeline):

    @staticmethod
    def get_pipeline_key_from_inputs(pipeline_key, model_path, text_checkpoint_dir, vae_checkpoint_dir):
        if pipeline_key is not None:
            return pipeline_key
        if model_path is None:
            raise ValueError(f"model_path {model_path} must be provided when pipeline_key is None")
        if isinstance(model_path, str) and not Path(model_path).exists():
            raise ValueError(f"model_path {model_path} does not exist")
        path = None
        if isinstance(model_path, (list, tuple)):
            path = text_checkpoint_dir or vae_checkpoint_dir
        return get_model_key_from_path(model_path if path is None else text_checkpoint_dir)

    @staticmethod
    def from_models(
        model_path,
        *,
        model_config: KsanaModelConfig = None,
        dist_config: KsanaDistributedConfig = None,
        pipeline_key: KsanaModelKey = None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_config: None | KsanaLoraConfig | list[KsanaLoraConfig] = None,
        offload_device="cpu",
    ) -> list[ModelBase]:
        log.info(f"Loading models from {model_path}")
        pipeline_key = KsanaPipeline.get_pipeline_key_from_inputs(
            pipeline_key, model_path, text_checkpoint_dir, vae_checkpoint_dir
        )
        model_config = model_config or KsanaModelConfig()
        dist_config = dist_config or KsanaDistributedConfig()
        engine = get_engine(dist_config=dist_config, offload_device=offload_device)

        # maybe cloud create pipeline as registered factory way with pipeline_key
        pipeline = KsanaPipeline(pipeline_key, engine, offload_device)
        pipeline.load_models(
            model_path,
            model_config=model_config,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora_config=lora_config,
        )
        return pipeline

    def load_models(
        self,
        model_path,
        *,
        model_config: KsanaModelConfig = None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_config: None | KsanaLoraConfig | list[KsanaLoraConfig] = None,
    ) -> list[ModelBase]:
        self.has_lora = lora_config is not None
        self.default_settings = load_default_settings(self.pipeline_key, with_lora=self.has_lora)
        load_model_path, text_checkpoint_dir, vae_checkpoint_dir = self._valid_input_models_path(
            model_path, text_checkpoint_dir, vae_checkpoint_dir, self.default_settings.diffusion
        )
        self.clear()

        # 1. load text encoder — V5: 通过 engine.run_loader_node
        self.text_encoder_key = self._get_text_encoder_key_from_pipeline_key(self.pipeline_key)
        self.engine.run_loader_node(
            self.text_encoder_key,
            model_path=text_checkpoint_dir,
        )

        # 2. load diffusion model — V5: 通过 engine.run_loader_node
        list_of_loras_list = self._valid_input_lora(lora_config, self.default_settings.diffusion)
        self.diffusion_model_key = self.pipeline_key
        self.engine.run_loader_node(
            self.diffusion_model_key,
            model_path=load_model_path,
            lora_config=list_of_loras_list,
            model_config=model_config,
        )

        # 3. load vae model — V5: 通过 engine.run_loader_node
        self.vae_model_key = self._get_vae_model_key_from_pipeline_key(self.pipeline_key)
        self.engine.run_loader_node(
            self.vae_model_key,
            model_path=os.path.join(vae_checkpoint_dir, self.default_settings.vae.checkpoint),
        )

        # save same info for later use
        self.vae_z_dim = self.default_settings.vae.z_dim
        self.vae_stride = self.default_settings.vae.stride
        self.patch_size = self.default_settings.diffusion.patch_size
        self.vae_scale_factor = getattr(self.default_settings.vae, "scale_factor", None)

    def _compute_noise_shape(self, *, target_w: int, target_h: int, target_f: int) -> list[int]:
        """根据 VAE 配置计算 noise latent 的空间形状 ``[z_dim, lat_f, lat_h, lat_w]``。

        委托给 :func:`compute_video_latent_shape` / :func:`compute_image_latent_shape`，
        与 ``KsanaVAEModel.create_latent_shape`` 共享同一份计算逻辑。
        """
        if self.vae_scale_factor is not None:
            return list(
                compute_image_latent_shape(
                    z_dim=self.vae_z_dim,
                    target_h=target_h,
                    target_w=target_w,
                    vae_scale_factor=self.vae_scale_factor,
                    patch_size=self.patch_size,
                )
            )
        return list(
            compute_video_latent_shape(
                z_dim=self.vae_z_dim,
                target_f=target_f,
                target_h=target_h,
                target_w=target_w,
                vae_stride=list(self.vae_stride),
                vae_patch=list(self.patch_size),
            )
        )

    @time_range
    @report("local_generate")
    def generate(
        self,
        prompt: str | list[str],
        *,
        prompt_negative: str | list[str] = None,
        img_path: str | list[str] | list[list[str]] = None,  # 参考图路径（Edit 模式）
        start_img_path: str | list[str] = None,  # 起始图路径（I2V 模式）
        end_img_path: str | list[str] = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
        input_latent: torch.Tensor = None,
        video_control_config: KsanaVaceContext = None,
    ):
        num_prompts = self._get_num_prompts(prompt)
        if num_prompts == 0:
            raise ValueError("prompt must be str or list of str")
        sample_config = self._valid_sample_config(sample_config, self.default_settings.sample_config)
        runtime_config = self._valid_runtime_config(
            runtime_config, self.default_settings.runtime_config, num_prompts=num_prompts
        )
        cache_config = self._valid_cache_config(cache_config, getattr(self.default_settings, "cache", None))
        log.info(f"generate prompt: {prompt}")
        log.info(f"sample_config : {sample_config}")
        log.info(f"runtime_config : {runtime_config}")
        log.info(f"cache_config : {cache_config}")
        img_path = self._valid_ref_images(img_path, num_prompts)
        start_img_path = self._valid_images(start_img_path, num_prompts)
        end_img_path = self._valid_images(end_img_path, num_prompts)
        with_end_image = end_img_path is not None

        vace_video_control_config = self._valid_video_control_config(video_control_config, runtime_config)

        target_frame_num = (
            vace_video_control_config.adjusted_frame_num
            if vace_video_control_config and vace_video_control_config.adjusted_frame_num
            else runtime_config.frame_num
        )

        # ── V5: 使用 tensor_scope + run_infer_node 编排 ──────────────
        with self.engine.tensor_scope():
            # 1. Text Encode
            condition_images = img_path if self.model_key == KsanaModelKey.QwenImage_Edit else None
            text_ctx = KsanaNodeContext(
                prompt=prompt,
                negative_prompt=prompt_negative,
                metadata={
                    "text_run_device": torch.device("cpu"),
                    "offload_model": runtime_config.offload_model,
                    "condition_images": condition_images,
                },
            )
            self.engine.run_infer_node(KsanaInferNodeType.TEXT_ENCODE, self.text_encoder_key, text_ctx)

            # 2. VAE Encode（如果有图像输入）
            if img_path is not None:
                # Edit 模式：参考图走 VAE_ENCODE_IMAGES — tensor 通过 put_tensors 写入
                img_tensor = self._load_input_images(img_path, None, device=self.offload_device)[0]
                self.engine.put_tensors(**{KsanaTensorKey.IMAGE: img_tensor})
                vae_ctx = KsanaNodeContext()
                self.engine.run_infer_node(KsanaInferNodeType.VAE_ENCODE_IMAGES, self.vae_model_key, vae_ctx)
            elif start_img_path is not None:
                # I2V 模式：起始图 + 结束图走 VAE_ENCODE_SPATIAL — tensor 通过 put_tensors 写入
                img_tensor, end_img_tensor = self._load_input_images(
                    start_img_path, end_img_path, device=self.offload_device
                )
                self.engine.put_tensors(
                    **{KsanaTensorKey.START_IMG: img_tensor, KsanaTensorKey.END_IMG: end_img_tensor}
                )
                vae_ctx = KsanaNodeContext(
                    metadata={
                        "target_f": target_frame_num,
                        "target_h": runtime_config.size[1],
                        "target_w": runtime_config.size[0],
                    }
                )
                self.engine.run_infer_node(KsanaInferNodeType.VAE_ENCODE_SPATIAL, self.vae_model_key, vae_ctx)

            # 3. Generator — 从 tensor_pool 读取 positive/negative/image_embeds
            #    有图像输入时 noise_shape 留 None，由 GeneratorNode 从 image_embeds 推导（与 V4 一致）；
            #    纯文生图/文生视频时由 Pipeline 显式计算。
            noise_shape = (
                None
                if start_img_path is not None
                else self._compute_noise_shape(
                    target_w=runtime_config.size[0], target_h=runtime_config.size[1], target_f=target_frame_num
                )
            )

            # input_latent 通过 put_tensors 写入 tensor_pool
            if input_latent is not None:
                self.engine.put_tensors(**{KsanaTensorKey.INPUT_LATENT: input_latent})

            gen_ctx = KsanaNodeContext(
                sample_config=sample_config,
                runtime_config=runtime_config,
                cache_config=cache_config,
                metadata={
                    "noise_shape": noise_shape,
                    "control_video_config": vace_video_control_config,
                },
            )
            self.engine.run_infer_node(KsanaInferNodeType.GENERATE, self.diffusion_model_key, gen_ctx)

            # 4. VAE Decode — 从 tensor_pool 读取 latents
            decode_ctx = KsanaNodeContext(
                metadata={
                    "offload_model": runtime_config.offload_model,
                    "with_end_image": with_end_image,
                },
            )
            self.engine.run_infer_node(KsanaInferNodeType.VAE_DECODE, self.vae_model_key, decode_ctx)
            outputs = self.engine.get_tensor(KsanaTensorKey.VIDEO)

        if runtime_config.offload_model:
            gc.collect()
            torch.cuda.synchronize()

        if outputs is not None and runtime_config.save_output:
            if self.pipeline_key.is_image_type():
                self._save_outputs(outputs, prompt, self.has_lora, runtime_config, save_image, ".png")
            else:
                self._save_outputs(outputs, prompt, self.has_lora, runtime_config, self._save_one_video, ".mp4")

        return outputs if runtime_config.return_frames else None
