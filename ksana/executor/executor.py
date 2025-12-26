from abc import ABC

import torch
import os
import gc
from datetime import datetime
from dataclasses import asdict
import torch.distributed as dist
from functools import partial
from PIL import Image

from ..models import KsanaDiffusionModel
from ..utils import log, time_range, save_video, merge_video_audio, is_dir
from ..config import (
    KsanaDistributedConfig,
    KsanaSampleConfig,
    KsanaRuntimeConfig,
    KsanaModelConfig,
    KsanaPipelineConfig,
)

from ..utils.logger import init_logging

from ..pipelines import create_ksana_pipeline

from ..distributed import shard_model
from ..models.model_pool import KsanaModelPool
from ..models.model_key import KsanaModelKey
from ..models import KsanaVAE
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig
import torchvision.transforms.functional as tvtf


class KsanaExecutor(ABC):
    """
    Base class for all Ksana executors.
    和模型有关的配置信息不放在Executor中，而是放在KsanaPipeline中
    """

    def __init__(self, device_id: int = 0, offload_device: str = "cpu"):
        """
        Initialize the executor.
        """
        self.device_id = device_id
        self.rank_id = device_id
        self.world_size = 1
        self.device = torch.device(f"cuda:{self.device_id}")
        self.offload_device = torch.device(offload_device)
        torch.cuda.set_device(self.device)
        self.pipeline = None
        # Note: each executor has its own model pool
        self.model_pool = KsanaModelPool()
        self.shard_fn = None
        self.dist_config = KsanaDistributedConfig(num_gpus=1, use_sp=False, dit_fsdp=False, ulysses_size=1)
        log.info(f"create executor with device_id {self.device_id}, offload_device {self.offload_device}")
        init_logging()

    def init_torch_dist_group(self, rank_id, dist_config: KsanaDistributedConfig):
        """r initialize sequence parallel group."""
        self.dist_config = dist_config
        log.info(f"init torch dist group with dist_config {dist_config}")
        if dist_config.num_gpus <= 1:
            return
        self.rank_id = rank_id
        self.world_size = dist_config.num_gpus
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank_id,
                device_id=self.device,
                world_size=dist_config.num_gpus,
            )
        log.info(f"init distributed group with rank_id {self.rank_id}, world_size {self.world_size}")
        init_logging(rank_id)
        self.shard_fn = partial(shard_model, device_id=self.device_id) if self.dist_config.dit_fsdp else None

    def create_pipeline(self, dir, model_config: KsanaModelConfig = None):
        if self.pipeline is not None:
            return self.pipeline
        model_name = os.path.basename(dir)
        model_name, task_type, model_size = KsanaDiffusionModel.get_model_type(model_name)
        default_pipeline_config = KsanaDiffusionModel.get_default_pipeline_config(model_name, task_type, model_size)
        pipeline_config = KsanaPipelineConfig(
            model_name=model_name,
            task_type=task_type,
            model_size=model_size,
            default_config=default_pipeline_config,
            model_config=model_config,
        )
        return create_ksana_pipeline(pipeline_config)

    def clear_models(self):
        """
        Clean models loaded by this executor.
        """
        if self.pipeline is None:
            return
        self.model_pool.clear()
        self.pipeline.clear_models()
        self.pipeline = None

    def load_models(
        self,
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora: None | str | list[list[dict], list[dict]] = None,
        model_config: KsanaModelConfig = None,
        **kwargs,
    ):
        self.clear_models()
        if not is_dir(model_path) and not isinstance(model_path, (list, tuple)):
            raise ValueError(f"model_path {model_path} is not exist, or not a directory")
        if isinstance(model_path, (list, tuple)):
            if text_checkpoint_dir is None:
                raise ValueError(
                    f"text_checkpoint_dir must be provided when loading from local checkpoint with diffusion model {model_path}"
                )
            if vae_checkpoint_dir is None:
                raise ValueError(
                    f"vae_checkpoint_dir must be provided when loading from local checkpoint with diffusion model {model_path}"
                )
        if len(kwargs) > 0:
            log.warning(f"kwargs {kwargs} are not used")
        # TODO: find a better way to get model_type, task_type, model_size
        self.pipeline = self.create_pipeline(
            dir=model_path if text_checkpoint_dir is None else text_checkpoint_dir, model_config=model_config
        )
        load_to_deivce = self.device if self.dist_config.num_gpus > 1 else self.offload_device
        model_list = self.pipeline.load_models(
            model_path=model_path,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora=lora,
            model_config=model_config,
            dist_config=self.dist_config,
            shard_fn=self.shard_fn,
            device=load_to_deivce,
            offload_device=self.offload_device,
        )
        self.model_pool.update_models(model_list)

    def load_diffusion_model(
        self,
        model_path,
        *,
        lora: None | str | list[list[dict], list[dict]] = None,
        model_config: KsanaModelConfig = None,
        comfy_bar_callback=None,
        **kwargs,
    ):
        self.clear_models()
        if len(kwargs) > 0:
            log.warning(f"kwargs {kwargs} are not used")
        dir_path = model_path
        if isinstance(model_path, (list, tuple)):
            # only create one pipeline for both model
            dir_path = model_path[0]
        self.pipeline = self.create_pipeline(dir=dir_path, model_config=model_config)
        load_to_deivce = self.device if self.dist_config.num_gpus > 1 else self.offload_device
        diffusion_model_list = self.pipeline.load_diffusion_model(
            model_path=model_path,
            lora=lora,
            model_config=model_config,
            dist_config=self.dist_config,
            device=load_to_deivce,
            offload_device=self.offload_device,
            shard_fn=self.shard_fn,
            comfy_bar_callback=comfy_bar_callback,
        )
        self.model_pool.update_models(diffusion_model_list)
        diffusion_model_key_list = [one_model.get_model_key() for one_model in diffusion_model_list]
        return diffusion_model_key_list

    def load_vae_model(self, model_path, allow_exist=False, **kwargs) -> KsanaModelKey:
        if len(kwargs) > 0:
            log.warning(f"kwargs {kwargs} are not in used")
        vae = KsanaVAE(
            model_path=model_path,
            device=self.offload_device,
        )
        self.model_pool.update_model(vae, allow_exist=allow_exist)
        return vae.get_model_key()

    def load_image(self, img_paths: list[str], target_len, device) -> torch.Tensor:
        if img_paths is None:
            return None
        log.info(f"load input image: {img_paths}")
        if len(img_paths) != 1 and len(img_paths) != target_len:
            raise ValueError(
                f"img_path length ({len(img_paths)}) must match prompt list length ({target_len}) or only one image"
            )
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

    def images_to_latents(
        self,
        img_path,
        end_img_path,
        prompts_list_len: int,
        runtime_config: KsanaRuntimeConfig,
    ):
        if not isinstance(img_path, (list, tuple)):
            img_path = [img_path]
        img_batch = self.load_image(img_path, prompts_list_len, device=self.offload_device)
        if end_img_path is not None:
            if not isinstance(end_img_path, (list, tuple)):
                end_img_path = [end_img_path]
            if len(end_img_path) != len(img_path):
                raise ValueError(
                    f"end_img_path length ({len(end_img_path)}) must match start_img_path length ({len(img_path)})"
                )
            end_img_batch = self.load_image(end_img_path, prompts_list_len, device=self.offload_device)
        else:
            end_img_batch = None

        vae_model = self.model_pool.get_model(self.pipeline.vae_key)

        img_latents = vae_model.forward_encode(
            target_f=runtime_config.frame_num,
            target_h=runtime_config.size[1],
            target_w=runtime_config.size[0],
            device=self.device,
            start_img=img_batch,
            end_img=end_img_batch,
            target_batch_size=prompts_list_len,
            vae_stride=self.pipeline.vae_stride,
            vae_patch=self.pipeline.patch_size,
        )

        return img_latents

    def valid_prompts(self, prompt, target_len=None):
        if prompt is None:
            return None
        if isinstance(prompt, str):
            prompts = [prompt]
        elif isinstance(prompt, (list, tuple)):
            prompts = list(prompt)
        else:
            raise TypeError(f"prompt must be str or list[str], got {type(prompt)}")
        if len(prompts) == 0:
            raise ValueError("prompt must not be empty")
        if target_len is not None:
            if len(prompts) == 1:
                prompts = prompts * target_len
            elif len(prompts) != target_len:
                raise ValueError(f"prompt length ({len(prompts)}) must match target length ({target_len})")
        return prompts

    def valid_sample_config(self, sample_config: KsanaSampleConfig, num_prompts):
        config_to_modify = sample_config if sample_config else KsanaSampleConfig()
        # Convert the frozen dataclass to a mutable dictionary
        config_dict = asdict(config_to_modify)
        # valid: batch_per_prompt to list
        batch_per_prompt = config_dict.get("batch_per_prompt")
        if batch_per_prompt is None:
            batch_per_prompt = [1] * num_prompts
        elif isinstance(batch_per_prompt, int):
            batch_per_prompt = [batch_per_prompt] * num_prompts
        elif isinstance(batch_per_prompt, (list, tuple)):
            if len(batch_per_prompt) != num_prompts:
                raise ValueError(
                    f"len(batch_per_prompt) ({len(batch_per_prompt)}) must match num_prompts ({num_prompts})"
                )
        else:
            raise TypeError(f"batch_per_prompt must be int|list[int]|None, got {type(batch_per_prompt)}")
        config_dict["batch_per_prompt"] = batch_per_prompt

        return KsanaSampleConfig(**config_dict)

    @time_range
    def generate_video(
        self,
        prompt: str | list[str],
        *,
        img_path: str | list[str] = None,
        end_img_path: str | list[str] = None,
        prompt_negative: str | list[str] = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        cache_configs: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
    ):
        prompts_list = self.valid_prompts(prompt)
        prompts_negative_list = self.valid_prompts(prompt_negative, len(prompts_list))
        with_end_image = end_img_path is not None
        sample_config = self.valid_sample_config(sample_config, num_prompts=len(prompts_list))
        runtime_config = runtime_config if runtime_config else KsanaRuntimeConfig()

        text_run_device = torch.device("cpu")  # TODO: maybe run text on cuda self.device
        positive, negative = self.pipeline.forward_text_encoder(
            self.model_pool,
            prompts_list,
            prompts_negative=prompts_negative_list,
            device=text_run_device,
            offload_device=self.offload_device,
            offload_model=runtime_config.offload_model,
        )

        # encode img if have
        img_latents = None
        if img_path is not None:
            img_latents = self.images_to_latents(
                img_path=img_path,
                end_img_path=end_img_path,
                prompts_list_len=len(prompts_list),
                runtime_config=runtime_config,
            )

        if cache_configs is not None and not isinstance(cache_configs, list):
            cache_configs = [cache_configs]

        latents = self.pipeline.forward_diffusion_models(
            model_pool=self.model_pool,
            positive=positive,
            negative=negative,
            img_latents=img_latents,
            sample_config=sample_config,
            runtime_config=runtime_config,
            cache_configs=cache_configs,
            device=self.device,
            offload_device=self.offload_device,
        )
        # TODO: multi-cards vae
        vae_model = self.model_pool.get_model(self.pipeline.vae_key)
        videos = vae_model.forward_decode(
            latents=latents, local_rank=self.rank_id, device=self.device, with_end_image=with_end_image
        )
        del latents
        if runtime_config.offload_model:
            gc.collect()
            torch.cuda.synchronize()

        self.save_videos(videos, prompts_list, runtime_config, sample_config.batch_per_prompt)
        res = videos if self.rank_id == 0 and runtime_config.return_frames else None
        # videos shape [bs, ch:3, f, h, w]
        return {self.rank_id: res}

    @time_range
    def forward_vae_encode(
        self, vae_key, *, frame_num: int, width: int, height: int, start_image=None, end_image=None, mask=None
    ):
        vae = self.model_pool.get_model(vae_key)
        log.info(
            f"vae_encode with vae_key: {vae_key}, frame_num: {frame_num}, width: {width}, height: {height}, "
            f"start_image shape: {start_image.shape if start_image is not None else None}, "
            f"end_image shape: {end_image.shape if end_image is not None else None}, "
            f"mask shape: {mask.shape if mask is not None else None}"
        )
        if self.rank_id != 0:
            return {self.rank_id: None}

        latents = vae.forward_encode(
            target_f=frame_num,
            target_h=height,
            target_w=width,
            device=self.device,
            start_img=start_image,
            end_img=end_image,
            mask=mask,
            target_batch_size=1 if start_image is None else start_image.shape[0],
        )
        return {self.rank_id: latents}

    @time_range
    def forward_vae_decode(self, vae_key, latents, with_end_image: bool = False):
        vae = self.model_pool.get_model(vae_key)
        latents = vae.forward_decode(
            latents, local_rank=self.rank_id, device=self.device, with_end_image=with_end_image
        )
        return {self.rank_id: latents}

    @time_range
    def forward_diffusion_models_with_tensors(self, model_keys, positive, negative, **kwargs):
        diffusion_models = self.model_pool.get_models(
            model_keys if isinstance(model_keys, (list, tuple)) else [model_keys]
        )
        latents = self.pipeline.forward_diffusion_models_with_tensors(
            diffusion_models=diffusion_models,
            positive=positive,
            negative=negative,
            device=self.device,
            offload_device=self.offload_device,
            **kwargs,
        )
        # only resturn latents on rank 0, since all rank have the same latents
        res = latents if self.rank_id == 0 else None
        return {self.rank_id: res}

    def get_save_path(self, output_folder, out_size, prompt_text, save_id):
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt_text.replace(" ", "_").replace("/", "_")[:30]
        suffix = ".mp4"
        save_file = (
            f"{self.pipeline.save_name}_{self.dist_config.num_gpus}cards_{out_size[0]}x{out_size[1]}_{formatted_time}_{formatted_prompt}_{save_id}"
            + suffix
        )
        return os.path.join(output_folder, save_file)

    def save_one_video(self, video, save_file):
        if self.rank_id != 0:
            return
        log.info(f"Saving generated video to {save_file}")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=self.pipeline.pipeline_config.default_config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if "s2v" in self.pipeline.task_type:
            audio_path = "tts.wav"
            merge_video_audio(video_path=save_file, audio_path=audio_path)

    def save_videos(self, videos, prompts_list, runtime_config, batch_per_prompt: list[int]):
        if self.rank_id != 0 or not runtime_config.save_video:
            return
        out_size = (
            runtime_config.size
            if runtime_config.size is not None
            else self.pipeline.pipeline_config.default_config.get("size", (None, None))
        )
        video_idx = 0
        for i in range(len(batch_per_prompt)):
            for j in range(batch_per_prompt[i]):
                video = videos[video_idx]
                prompt_text = prompts_list[i]
                save_path = self.get_save_path(runtime_config.output_folder, out_size, prompt_text, j)
                self.save_one_video(video, save_path)
                video_idx += 1
