from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from ..config import KsanaDistributedConfig, KsanaModelConfig, KsanaRuntimeConfig, KsanaSampleConfig
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
from ..config.lora_config import KsanaLoraConfig
from ..engine import get_engine
from ..models.base_model import KsanaModel
from ..models.model_key import KsanaModelKey, get_model_key_from_path
from ..settings import load_default_settings
from ..units import KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range
from ..utils.media import save_image
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
        pipeline_key: KsanaModelKey = None,  # used model key as pipeline key now
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_config: None | KsanaLoraConfig | list[KsanaLoraConfig] = None,
        offload_device="cpu",
    ) -> list[KsanaModel]:
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
    ) -> list[KsanaModel]:
        self.has_lora = lora_config is not None
        self.default_settings = load_default_settings(self.pipeline_key, with_lora=self.has_lora)
        load_model_path, text_checkpoint_dir, vae_checkpoint_dir = self._valid_input_models_path(
            model_path, text_checkpoint_dir, vae_checkpoint_dir, self.default_settings.diffusion
        )
        self.clear()

        # 1. load text encoder
        # TODO: use load_text_encoder of engine and only save key in future
        text_model_key = self._get_text_encoder_key_from_pipeline_key(self.pipeline_key)
        text_model_loader = KsanaUnitFactory.create(KsanaUnitType.LOADER, text_model_key)
        # TODO(rock): support text encode on cuda later
        self.text_encoder_model = text_model_loader.run(checkpoint_dir=text_checkpoint_dir, device=self.offload_device)

        # 2. load diffusion model
        list_of_loras_list = self._valid_input_lora(lora_config, self.default_settings.diffusion)
        self.diffusion_model_key = self.engine.load_diffusion_model(
            load_model_path,
            model_key=self.model_key,
            lora_config=list_of_loras_list,
            model_config=model_config,
        )

        # 3. load vae model
        self.vae_model_key = self.engine.load_vae_model(
            os.path.join(vae_checkpoint_dir, self.default_settings.vae.checkpoint),
            model_key=self._get_vae_model_key_from_pipeline_key(self.pipeline_key),
        )

        # same same info for later use
        self.vae_z_dim = self.default_settings.vae.z_dim
        self.vae_stride = self.default_settings.vae.stride
        self.patch_size = self.default_settings.diffusion.patch_size

    @time_range
    def generate(
        self,
        prompt: str | list[str],
        *,
        prompt_negative: str | list[str] = None,
        img_path: str | list[str] = None,
        end_img_path: str | list[str] = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
    ):
        """local use for generate"""
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
        img_path = self._valid_images(img_path, num_prompts)
        end_img_path = self._valid_images(end_img_path, num_prompts)
        with_end_image = end_img_path is not None

        text_run_device = torch.device("cpu")  # TODO: maybe run text on cuda self.device
        text_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, self.text_encoder_model.model_key)
        positive, negative = text_encoder.run(
            self.text_encoder_model,
            prompts_positive=prompt,
            prompts_negative=prompt_negative,
            device=text_run_device,
            offload_device=self.offload_device,
            offload_model=runtime_config.offload_model,
        )

        img_tensor, end_img_tensor = self._load_input_images(img_path, end_img_path, device=self.offload_device)
        img_latents = self.engine.forward_vae_encode(
            model_key=self.vae_model_key,
            target_f=runtime_config.frame_num,
            target_h=runtime_config.size[1],
            target_w=runtime_config.size[0],
            start_img=img_tensor,
            end_img=end_img_tensor,
        )

        latents = self.engine.forward_generator(
            model_key=self.model_key,
            positive=positive,
            negative=negative,
            img_latents=img_latents,
            sample_config=sample_config,
            runtime_config=runtime_config,
            cache_config=cache_config,
        )
        del positive, negative, img_latents

        outputs = self.engine.forward_vae_decode(
            model_key=self.vae_model_key,
            latents=latents,
            offload_device=self.offload_device,
            offload_model=runtime_config.offload_model,
            with_end_image=with_end_image,
        )
        del latents
        if runtime_config.offload_model:
            gc.collect()
            torch.cuda.synchronize()

        if runtime_config.save_output:
            if len(outputs.shape) > 4:  # [B,C,F,H,W]
                self._save_outputs(outputs, prompt, self.has_lora, runtime_config, self._save_one_video, ".mp4")
            else:  # [B,C,H,W]
                self._save_outputs(outputs, prompt, self.has_lora, runtime_config, save_image, ".png")

        return outputs if runtime_config.return_frames else None
