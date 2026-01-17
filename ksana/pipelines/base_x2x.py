from __future__ import annotations

import gc
import os
from abc import ABC
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.functional as tvtf
from PIL import Image

from ..config import (
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
from ..engine import KsanaEngine, get_engine
from ..models import KsanaT5TextEncoderModel
from ..models.base_model import KsanaModel
from ..models.model_key import KsanaModelKey, get_model_key_from_path
from ..settings import load_default_settings
from ..units import KsanaUnitFactory, KsanaUnitType
from ..utils import log, merge_video_audio, time_range
from ..utils.lora import build_loras_list
from ..utils.media import save_image, save_video
from ..utils.types import evolve_with_recommend, str_to_list


class KsanaPipeline(ABC):
    def __init__(self, model_key: KsanaModelKey, engine: KsanaEngine, offload_device):
        self.pipeline_key = model_key
        self.engine = engine
        self.offload_device = offload_device
        self.default_settings = None

        # lora info for save name
        self.has_lora = False

        # save model keys
        self.vae_model_key = None
        self.diffusion_model_key = None
        # TODO(rock): use text_encoder key when support load_text_encoder node
        self.text_encoder_model = None

    @property
    def model_key(self) -> KsanaModelKey:
        return self.pipeline_key

    @property
    def pipeline_name(self):
        return f"{self.pipeline_key.name}"

    def clear(self):
        self.text_encoder_key = None
        self.vae_model_key = None
        self.diffusion_model_keys = None
        self.has_lora = False

    def _valid_sample_config(self, sample_config: KsanaSampleConfig, default_configs):
        sample_config = sample_config if sample_config else KsanaSampleConfig()
        recommend_configs = {
            "steps": getattr(default_configs, "steps", None),
            "cfg_scale": getattr(default_configs, "cfg_scale", None),
            "shift": getattr(default_configs, "shift", None),
            "solver": (
                KsanaSolverType(default_configs.solver)
                if isinstance(getattr(default_configs, "solver", None), str)
                else None
            ),
            "denoise": getattr(default_configs, "denoise", None),
        }
        sample_config = evolve_with_recommend(sample_config, recommend_configs)
        return sample_config

    def _valid_runtime_config(self, runtime_config: KsanaRuntimeConfig, default_configs, num_prompts: int):
        runtime_config = runtime_config or KsanaRuntimeConfig()
        # valid: batch_size_per_prompts to list
        batch_size_per_prompts = runtime_config.batch_size_per_prompts
        if batch_size_per_prompts is None:
            batch_size_per_prompts = [1] * num_prompts
        elif isinstance(batch_size_per_prompts, int):
            batch_size_per_prompts = [batch_size_per_prompts] * num_prompts
        elif isinstance(batch_size_per_prompts, (list, tuple)):
            if len(batch_size_per_prompts) != num_prompts:
                raise ValueError(
                    f"batch_size_per_prompts({batch_size_per_prompts}) len must match num_prompts ({num_prompts})"
                )
        else:
            raise TypeError(
                f"batch_size_per_prompts must be int/list[int]/None, but got {type(batch_size_per_prompts)}"
            )
        runtime_config = evolve_with_recommend(
            runtime_config,
            {"batch_size_per_prompts": batch_size_per_prompts},
            force_update=True,
        )
        recommend_configs = {
            "size": getattr(default_configs, "target_size", None),
            "frame_num": getattr(default_configs, "frame_num", None),
        }
        runtime_config = evolve_with_recommend(
            runtime_config,
            recommend_configs,
            force_update=False,
        )
        return runtime_config

    # TODO(TJ): use cache yamls
    def _valid_cache_config(self, cache_config: KsanaCacheConfig, default_configs):  # pylint: disable=unused-argument
        if cache_config is None:
            return None
        return [cache_config] if cache_config is not isinstance(cache_config, list) else cache_config

    def _valid_images(self, img_path, prompts_list_len: int):
        if img_path is None:
            return None
        img_path = str_to_list(img_path)
        if len(img_path) != 1 and len(img_path) != prompts_list_len:
            raise ValueError(
                f"img_path length ({len(img_path)}) must match prompt list length ({prompts_list_len}) "
                "or only one image"
            )
        return img_path

    def _save_one_video(self, video, save_path, is_s2v=False):
        save_video(
            tensor=video[None],
            save_file=save_path,
            fps=self.default_settings.sample_config.fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if is_s2v:
            audio_path = "tts.wav"
            merge_video_audio(video_path=save_path, audio_path=audio_path)

    def _get_save_filename(
        self, out_size: list[int], prompt_text: str, has_lora: bool, num_cards: int, batch_id: int, suffix=".mp4"
    ):
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt_text.replace(" ", "_").replace("/", "_")[:30]
        lora_str = "_with_lora" if has_lora else ""
        return (
            f"{self.pipeline_name}_{num_cards}cards_w{out_size[0]}_h{out_size[1]}{lora_str}"
            + f"_{formatted_time}_{formatted_prompt}_{batch_id}{suffix}"
        )

    def _save_outputs(
        self, outputs, input_prompt: str | list[str], has_lora: bool, runtime_config, save_one_func, suffix: str
    ):
        input_prompt = str_to_list(input_prompt)
        out_size = runtime_config.size
        output_idx = 0
        prompt_len = len(input_prompt)
        batch_size_per_prompts = runtime_config.batch_size_per_prompts
        if prompt_len != len(batch_size_per_prompts):
            raise RuntimeError(
                f"len({prompt_len}) of input_prompt({input_prompt}) "
                f"must match len({len(batch_size_per_prompts)}) of batch_size_per_prompts({batch_size_per_prompts})."
            )
        num_cards = self.engine.num_gpus
        for prmopt_id in range(prompt_len):
            for j in range(batch_size_per_prompts[prmopt_id]):
                prompt_text = input_prompt[prmopt_id]
                output = outputs[output_idx]
                save_filename = self._get_save_filename(out_size, prompt_text, has_lora, num_cards, j, suffix=suffix)
                save_path = os.path.join(runtime_config.output_folder, save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                log.info(f"Saving generated image to {save_path}")
                save_one_func(output, save_path)
                output_idx += 1

    @staticmethod
    def _valid_input_model_paths(model_path, text_checkpoint_dir, vae_checkpoint_dir):
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
        elif Path(model_path).is_dir():
            text_checkpoint_dir = text_checkpoint_dir or model_path
            vae_checkpoint_dir = vae_checkpoint_dir or model_path
        else:
            raise ValueError(f"model_path {model_path} should be a directory or list of diffusion model files")
        return model_path, text_checkpoint_dir, vae_checkpoint_dir

    # TODO(TJ): optimize me with _valid_input_model_paths
    def _valid_input_models_path(self, model_path, diffusion_default_settings):
        load_model_path = model_path
        if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
            load_model_path = [
                os.path.join(model_path, diffusion_default_settings.high_noise_checkpoint),
                os.path.join(model_path, diffusion_default_settings.low_noise_checkpoint),
            ]
        else:
            load_model_path = model_path
        return load_model_path

    @staticmethod
    def from_models(
        model_path,
        *,
        model_config: KsanaModelConfig = None,
        dist_config: KsanaDistributedConfig = None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora: None | str | list[list[dict], list[dict]] = None,
        offload_device="cpu",
    ) -> list[KsanaModel]:
        log.info(f"Loading models from {model_path}")
        model_path, text_checkpoint_dir, vae_checkpoint_dir = KsanaPipeline._valid_input_model_paths(
            model_path, text_checkpoint_dir, vae_checkpoint_dir
        )
        model_config = model_config or KsanaModelConfig()
        dist_config = dist_config or KsanaDistributedConfig()
        engine = get_engine(dist_config=dist_config, offload_device=offload_device)
        model_key = get_model_key_from_path(model_path if text_checkpoint_dir is None else text_checkpoint_dir)

        # maybe need create pipeline from registered factory
        pipeline = KsanaPipeline(model_key, engine, offload_device)
        pipeline.load_models(
            model_path,
            model_config=model_config,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora=lora,
        )
        return pipeline

    def _load_text_encoder(self, text_checkpoint_dir, default_text_settings):
        if self.pipeline_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
            text_encoder = KsanaT5TextEncoderModel(
                model_key=KsanaModelKey.T5TextEncoder,
                default_settings=default_text_settings,
                checkpoint_path=os.path.join(text_checkpoint_dir, default_text_settings.checkpoint),
                tokenizer_path=os.path.join(text_checkpoint_dir, default_text_settings.tokenizer),
                dtype=default_text_settings.dtype,
                device=torch.device("cpu"),
            )
        else:
            raise ValueError(f"text_encoder {self.pipeline_key} not supported in pipeline")
        if self.offload_device:
            text_encoder.to(self.offload_device)
        return text_encoder

    def _valid_input_lora(self, lora: str | list[str], diffusion_default_settings):
        if lora is None:
            return None
        list_of_loras_list = None
        if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
            if Path(lora).is_dir():
                lora_dir = lora
                list_of_loras_list = []
                list_of_loras_list.append(
                    build_loras_list(os.path.join(lora_dir, diffusion_default_settings.high_noise_lora_checkpoint))
                )
                list_of_loras_list.append(
                    build_loras_list(os.path.join(lora_dir, diffusion_default_settings.low_noise_lora_checkpoint))
                )
            else:
                raise ValueError(f"lora {lora} must be a directory in {self.model_key}")
        else:
            raise NotImplementedError(f"lora {lora} not supported in pipeline {self.model_key} yet")
        return list_of_loras_list

    def load_models(
        self,
        model_path,
        *,
        model_config: KsanaModelConfig = None,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora: None | str | list[list[dict], list[dict]] = None,
    ) -> list[KsanaModel]:
        self.engine.clear_models()
        # keep lora flag for output name
        self.has_lora = lora is not None
        self.default_settings = load_default_settings(self.pipeline_key)

        # TODO: use load_text_encoder in engine in future
        self.text_encoder_model = self._load_text_encoder(text_checkpoint_dir, self.default_settings.text_encoder)

        load_model_path = self._valid_input_models_path(model_path, self.default_settings.diffusion)
        list_of_loras_list = self._valid_input_lora(lora, self.default_settings.diffusion)
        self.diffusion_model_key = self.engine.load_diffusion_model(
            load_model_path,
            model_key=self.model_key,
            lora=list_of_loras_list,
            model_config=model_config,
        )
        self.vae_model_key = self.engine.load_vae_model(
            os.path.join(vae_checkpoint_dir, self.default_settings.vae.checkpoint),
        )

        # same same info for later use
        self.vae_z_dim = self.default_settings.vae.z_dim
        self.vae_stride = self.default_settings.vae.stride
        self.patch_size = self.default_settings.diffusion.patch_size

    def _get_num_prompts(self, prompt: str | list[str]):
        if prompt is None:
            return 0
        if isinstance(prompt, str):
            return 1
        elif isinstance(prompt, list):
            return len(prompt)
        else:
            raise ValueError(f"prompt {prompt} must be str or list of str")

    def _load_image(self, img_paths: list[str], device) -> torch.Tensor:
        # move to utils
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
                # Note: if img is a list, then all image shapes must be the same
                # otherwise the latents shape are not equal for batching
                raise ValueError(f"all images {img_paths} should have the same shape, but got {img.size} and {shape}")
            img = tvtf.to_tensor(img).sub_(0.5).div_(0.5).to(device)
            imgs.append(img.unsqueeze(0))
        if len(imgs) == 1:
            return imgs[0]
        else:
            return torch.cat(imgs, dim=0)

    def _load_input_images(self, img_path: str | list[str], end_img_path: str | list[str], device):
        img_tensor = self._load_image(img_path, device=device)
        end_img_tensor = self._load_image(end_img_path, device=device)
        if end_img_path is not None and img_path is None:
            raise ValueError(f"img_path must be not None when end_img_path {end_img_path} is not None")
        return img_tensor, end_img_tensor

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
            cache_configs=cache_config,
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
