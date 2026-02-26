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

import os
from abc import ABC
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.functional as tvtf
from omegaconf import OmegaConf
from PIL import Image

from ..config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
from ..config.lora_config import KsanaLoraConfig
from ..engine import KsanaEngine
from ..models.model_key import KsanaModelKey
from ..utils import log, merge_video_audio
from ..utils.media import save_video
from ..utils.types import evolve_with_recommend, str_to_list
from ..utils.vace import KsanaVaceContext, build_vace_video_control_config, latent_process_out


class KsanaBasePipeline(ABC):
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
        if self.engine:
            self.engine.clear_models()
        self.vae_model_key = None
        self.diffusion_model_key = None
        self.text_encoder_model = None
        self.has_lora = False

    def _valid_sample_config(self, sample_config: KsanaSampleConfig, default_configs):
        sample_config = sample_config if sample_config else KsanaSampleConfig()
        cfg_scale = getattr(default_configs, "cfg_scale", None)
        cfg_scale = OmegaConf.to_container(cfg_scale, resolve=True) if OmegaConf.is_list(cfg_scale) else cfg_scale
        solver = getattr(default_configs, "solver", None)
        solver = KsanaSolverType(solver) if isinstance(solver, str) else solver
        recommend_configs = {
            "steps": getattr(default_configs, "steps", None),
            "shift": getattr(default_configs, "shift", None),
            "denoise": getattr(default_configs, "denoise", None),
            "cfg_scale": cfg_scale,
            "solver": solver,
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
        if isinstance(cache_config, (tuple, list)):
            return list(cache_config)
        elif isinstance(cache_config, (KsanaHybridCacheConfig, KsanaCacheConfig)):
            return [cache_config]
        else:
            raise ValueError(
                f"cache_config must be KsanaHybridCacheConfig or KsanaCacheConfig, but got {type(cache_config)}"
            )

    def _valid_video_control_config(
        self,
        video_control_config: KsanaVaceContext | None,
        runtime_config: KsanaRuntimeConfig,
    ) -> KsanaVaceContext | None:
        width, height = runtime_config.size
        num_frames = runtime_config.frame_num

        def vae_encode_fn(frame: torch.Tensor) -> torch.Tensor:
            latents = self.engine.forward_vae_encode_image(model_key=self.vae_model_key, image=frame)
            return latent_process_out(latents)

        vace_config = build_vace_video_control_config(
            video_control_config=video_control_config,
            width=width,
            height=height,
            num_frames=num_frames,
            vae_encode_fn=vae_encode_fn,
        )

        if vace_config is not None and vace_config.trim_latent > 0:
            vae_temporal_stride = self.vae_stride[0]  # typically 4
            adjusted_frame_num = num_frames + vace_config.trim_latent * vae_temporal_stride
            vace_config.adjusted_frame_num = adjusted_frame_num
        elif vace_config is not None:
            vace_config.adjusted_frame_num = num_frames

        return vace_config

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

    def _get_text_encoder_key_from_pipeline_key(self, pipeline_key):
        pipeline_key_to_text_encoder_key = {
            KsanaModelKey.Wan2_2_I2V_14B: KsanaModelKey.T5TextEncoder,
            KsanaModelKey.Wan2_2_T2V_14B: KsanaModelKey.T5TextEncoder,
            KsanaModelKey.Wan2_1_VACE_14B: KsanaModelKey.T5TextEncoder,
            KsanaModelKey.QwenImage_T2I: KsanaModelKey.Qwen2VLTextEncoder,
        }
        text_encoder_key = pipeline_key_to_text_encoder_key.get(pipeline_key, None)
        if text_encoder_key is None:
            raise ValueError(f"text_encoder for pipeline {pipeline_key} not supported yet")
        return text_encoder_key

    def _get_vae_model_key_from_pipeline_key(self, pipeline_key):
        pipeline_key_to_vae_model_key = {
            KsanaModelKey.Wan2_2_I2V_14B: KsanaModelKey.VAE_WAN2_1,
            KsanaModelKey.Wan2_2_T2V_14B: KsanaModelKey.VAE_WAN2_1,
            KsanaModelKey.Wan2_1_VACE_14B: KsanaModelKey.VAE_WAN2_1,
            KsanaModelKey.Wan2_2_TI2V_5B: KsanaModelKey.VAE_WAN2_2,
            KsanaModelKey.QwenImage_T2I: KsanaModelKey.QwenImageVAE,
        }
        vae_model_key = pipeline_key_to_vae_model_key.get(pipeline_key, None)
        if vae_model_key is None:
            raise ValueError(f"vae_model for pipeline {pipeline_key} not supported yet")
        return vae_model_key

    def _valid_input_lora(self, lora_config: KsanaLoraConfig | list[KsanaLoraConfig], diffusion_default_settings):
        if lora_config is None:
            return None
        lora_list = []
        if isinstance(lora_config, KsanaLoraConfig):
            lora_list = [lora_config]
        elif isinstance(lora_config, (list, tuple)):
            lora_list = list(lora_config)
        else:
            raise ValueError(f"lora_config {lora_config} must be a KsanaLoraConfig or a list of KsanaLoraConfig")
        list_of_loras_list = []
        if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
            lora_list_high = []
            lora_list_low = []
            for one_lora in lora_list:
                if not isinstance(one_lora, KsanaLoraConfig):
                    raise ValueError(f"one_lora {one_lora} must be a KsanaLoraConfig")
                if not Path(one_lora.path).is_dir():
                    raise ValueError(f"one_lora.path {one_lora.path} must be a directory for {self.model_key}")
                lora_list_high.append(
                    KsanaLoraConfig(
                        path=os.path.join(one_lora.path, diffusion_default_settings.high_noise_lora_checkpoint),
                        strength=one_lora.strength,
                    )
                )
                lora_list_low.append(
                    KsanaLoraConfig(
                        path=os.path.join(one_lora.path, diffusion_default_settings.low_noise_lora_checkpoint),
                        strength=one_lora.strength,
                    )
                )
            # [high: multi lora list[KsanaLoraConfig], low: multi lora list[KsanaLoraConfig]]
            list_of_loras_list = [lora_list_high, lora_list_low]
        else:
            list_of_loras_list = [lora_list]
        return list_of_loras_list

    def _valid_input_models_path(self, model_path, text_checkpoint_dir, vae_checkpoint_dir, diffusion_default_settings):
        load_model_path = model_path
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
            load_model_path = list(model_path)
        elif Path(model_path).is_dir():
            text_checkpoint_dir = text_checkpoint_dir or model_path
            vae_checkpoint_dir = vae_checkpoint_dir or model_path
            if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
                load_model_path = [
                    os.path.join(model_path, diffusion_default_settings.high_noise_checkpoint),
                    os.path.join(model_path, diffusion_default_settings.low_noise_checkpoint),
                ]
        else:
            raise ValueError(f"model_path {model_path} should be a directory or list of diffusion model files")
        return load_model_path, text_checkpoint_dir, vae_checkpoint_dir

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
