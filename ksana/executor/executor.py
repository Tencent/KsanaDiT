from abc import ABC

import torch
import os

from datetime import datetime
import gc


from ..models import KsanaDiffusionModel
from ..utils import log, time_range, save_video, merge_video_audio
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

    def create_pipeline(self, dir, model_config: KsanaModelConfig = None, comfy_model_config=None):
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

    def load_models_from_pretrained(self, checkpoint_dir, lora_dir=None, model_config: KsanaModelConfig = None):
        self.pipeline = self.create_pipeline(dir=checkpoint_dir, model_config=model_config)
        # all model load to cpu, how about multi gpus?
        load_to_deivce = self.device if self.dist_config.use_sp else torch.device("cpu")  # TODO: check me
        self.pipeline.load_models_from_pretrained(
            checkpoint_dir=checkpoint_dir,
            lora_dir=lora_dir,
            model_config=model_config,
            dist_config=self.dist_config,
            shard_fn=self.shard_fn,
            device=load_to_deivce,
            offload_device=self.offload_device,
        )

    def load_diffusion_model_from_comfy(
        self,
        model_config: KsanaModelConfig,
        comfy_model_path: str = None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_operations=None,
    ):
        self.pipeline = self.create_pipeline(
            dir=comfy_model_path, model_config=model_config, comfy_model_config=comfy_model_config
        )

        load_to_deivce = self.device if self.dist_config.use_sp else torch.device("cpu")  # TODO: check me
        return self.pipeline.load_diffusion_model_from_comfy(
            model_config=model_config,
            dist_config=self.dist_config,
            comfy_model_path=comfy_model_path,
            comfy_model_config=comfy_model_config,
            comfy_model_state_dict=comfy_model_state_dict,
            comfy_operations=comfy_operations,
            shard_fn=self.shard_fn,
            device=load_to_deivce,
            offload_device=self.offload_device,
        )

    @time_range
    def generate_video(
        self,
        prompt: str,
        prompt_negative: str = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
    ):
        sample_config = sample_config if sample_config else KsanaSampleConfig()
        runtime_config = runtime_config if runtime_config else KsanaRuntimeConfig()
        text_run_device = torch.device("cpu")  # TODO: maybe run text on cuda self.device
        positive, negative = self.pipeline.forward_text_encoder(
            prompt, prompt_negative, device=text_run_device, offload_device=self.offload_device
        )
        latents = self.pipeline.forward_diffusion_model(
            positive,
            negative,
            sample_config=sample_config,
            runtime_config=runtime_config,
            device=self.device,
            offload_device=self.offload_device,
        )
        if runtime_config.offload_model:
            self.pipeline.offload_diffusion_model_to(self.offload_device)
        videos = self.pipeline.forward_vae(
            latents, local_rank=self.local_rank, device=self.device, offload_device=self.offload_device
        )

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
