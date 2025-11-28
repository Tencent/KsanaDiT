from abc import ABC

import torch
import os

from datetime import datetime
import gc


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
import torch.distributed as dist

from ..pipelines import create_ksana_pipeline

from functools import partial
from ..distributed import shard_model


class KsanaExecutor(ABC):
    """
    Base class for all Ksana executors.
    和模型有关的配置信息不放在Executor中，而是放在KsanaPipeline中
    """

    model = None

    def __init__(self, dist_config: KsanaDistributedConfig = None, offload_device: str = "cpu"):
        """
        Initialize the executor.
        """
        self.dist_config = dist_config
        self.local_rank = self.dist_config.local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.offload_device = torch.device(offload_device)

        init_logging(self.local_rank)
        if self.dist_config.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=self.dist_config.rank_id,
                world_size=self.dist_config.world_size,
            )

        self.shard_fn = partial(shard_model, device_id=self.local_rank) if self.dist_config.dit_fsdp else None
        self.pipeline = None

    def create_pipeline(self, dir, model_config: KsanaModelConfig = None, comfy_model_config=None):
        if self.pipeline is not None:
            return self.pipeline
        model_name = os.path.basename(dir)
        model_name, task_type, model_size = KsanaDiffusionModel.get_model_type(model_name, comfy_model_config)
        default_pipeline_config = KsanaDiffusionModel.get_default_pipeline_config(model_name, task_type, model_size)
        pipeline_config = KsanaPipelineConfig(
            model_name=model_name,
            task_type=task_type,
            model_size=model_size,
            default_config=default_pipeline_config,
            model_config=model_config,
        )
        return create_ksana_pipeline(pipeline_config)

    def load_models(
        self,
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        **kwargs,
    ):
        if isinstance(model_path, (list, tuple)) or not is_dir(model_path):
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
        load_to_deivce = self.device if self.dist_config.world_size > 1 else self.offload_device
        return self.pipeline.load_models(
            model_path=model_path,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora_dir=lora_dir,
            model_config=model_config,
            dist_config=self.dist_config,
            shard_fn=self.shard_fn,
            device=load_to_deivce,
            offload_device=self.offload_device,
        )

    def load_diffusion_model(
        self,
        model_path,
        *,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        comfy_model_config=None,
        comfy_model_state_dict=None,
        **kwargs,
    ) -> KsanaDiffusionModel | tuple[KsanaDiffusionModel, KsanaDiffusionModel]:
        if len(kwargs) > 0:
            log.warning(f"kwargs {kwargs} are not used")
        self.pipeline = self.create_pipeline(dir=model_path, model_config=model_config)
        load_to_deivce = self.device if self.dist_config.world_size > 1 else self.offload_device
        return self.pipeline.load_diffusion_model(
            model_path=model_path,
            lora_dir=lora_dir,
            model_config=model_config,
            dist_config=self.dist_config,
            comfy_model_config=comfy_model_config,
            comfy_model_state_dict=comfy_model_state_dict,
            device=load_to_deivce,
            offload_device=self.offload_device,
            shard_fn=self.shard_fn,
        )

    @time_range
    def generate_one_video(
        self,
        prompt: str,
        prompt_negative: str = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
    ):
        latents = self.pipeline.generate_video(
            prompt=prompt,
            prompt_negative=prompt_negative,
            sample_config=sample_config,
            runtime_config=runtime_config,
            device=self.device,
            offload_device=self.offload_device,
        )
        # TODO: multi-cards vae
        videos = self.pipeline.forward_vae(
            latents,
            local_rank=self.local_rank,
            device=self.device,
            offload_device=self.offload_device,
            offload_model=runtime_config.offload_model,
        )
        del latents
        if runtime_config.offload_model:
            gc.collect()
            torch.cuda.synchronize()
        # TODO: move save to generator, outside executors
        if runtime_config.save_video:
            self.save_video(videos, self.get_save_path(runtime_config.output_folder, prompt))

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if runtime_config.return_frames and self.dist_config.rank_id == 0:
            return videos

    @time_range
    def generate_video(
        self,
        prompt,
        prompt_negative: str = None,
        *,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
    ):
        sample_config = sample_config if sample_config else KsanaSampleConfig()
        runtime_config = runtime_config if runtime_config else KsanaRuntimeConfig()
        if isinstance(prompt, (list, tuple)):
            # TODO: support bs > 1 inside
            res = []
            for i in range(len(prompt)):
                one_prompt = prompt[i]
                one_negative = prompt_negative[i] if isinstance(prompt_negative, (tuple, list)) else prompt_negative
                res.append(
                    self.generate_one_video(
                        one_prompt,
                        prompt_negative=one_negative,
                        sample_config=sample_config,
                        runtime_config=runtime_config,
                    )
                )
            return res
        else:
            return self.generate_one_video(
                prompt, prompt_negative=prompt_negative, sample_config=sample_config, runtime_config=runtime_config
            )

    @time_range
    def generate_video_with_tensors(self, model, positive, negative, **kwargs):
        return self.pipeline.generate_video_with_tensors(
            model=model,
            positive=positive,
            negative=negative,
            device=self.device,
            offload_device=self.offload_device,
            **kwargs,
        )

    # TODO: move save to generator, outside executors
    def get_save_path(self, output_folder, prompt_text):
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt_text.replace(" ", "_").replace("/", "_")[:30]
        suffix = ".mp4"
        save_file = (
            f"{self.pipeline.save_name}_{self.dist_config.world_size}cards_{formatted_time}_{formatted_prompt}" + suffix
        )
        return os.path.join(output_folder, save_file)

    def save_video(self, video, save_file):
        if self.dist_config.rank_id != 0:
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
