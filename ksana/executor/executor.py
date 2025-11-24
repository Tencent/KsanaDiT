from abc import ABC

import torch
import os

from datetime import datetime
import gc


from ..models import KsanaDiffusionModel
from ..utils import log, time_range, save_video, merge_video_audio
from ..config import KsanaDistributedConfig

from ..utils.logger import init_logging
import torch.distributed as dist
from ..utils.const import DEFAULT_OUTPUTS_VIDEO_DIR

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

    def load_model_from_pretrained(self, checkpoint_dir, lora_dir=None, torch_compile_config=None):
        model_name = os.path.basename(checkpoint_dir)
        model_name, model_type, model_size = KsanaDiffusionModel.get_model_type(model_name)
        self.pipeline = create_ksana_pipeline(model_name, model_type, model_size)
        # all model load to cpu, how about multi gpus?
        load_to_deivce = self.device if self.dist_config.use_sp else torch.device("cpu")  # TODO: check me
        self.pipeline.load_model_from_pretrained(
            checkpoint_dir=checkpoint_dir,
            lora_dir=lora_dir,
            torch_compile_config=torch_compile_config,
            dist_config=self.dist_config,
            shard_fn=self.shard_fn,
            device=load_to_deivce,
            offload_device=self.offload_device,
        )

    @time_range
    def generate_video(
        self,
        prompt: str,
        prompt_negative: str = None,
        save_video: bool = True,
        return_frames: bool = False,
        output_folder: str = DEFAULT_OUTPUTS_VIDEO_DIR,
        offload_model: bool = False,
        **kwargs,
    ):
        text_run_device = torch.device("cpu")  # TODO: maybe run text on cuda self.device
        positive, negative = self.pipeline.forward_text_encoder(
            prompt, prompt_negative, device=text_run_device, offload_device=self.offload_device
        )
        latents = self.pipeline.forward_diffusion_model(
            positive, negative, device=self.device, offload_device=self.offload_device, **kwargs
        )
        if offload_model:
            self.pipeline.offload_diffusion_model_to(self.offload_device)
        videos = self.pipeline.forward_vae(
            latents, local_rank=self.local_rank, device=self.device, offload_device=self.offload_device
        )

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        # TODO: move save to generator, outside executors
        if save_video:
            self.save_video(videos, self.get_save_path(output_folder, prompt))

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        if return_frames and self.dist_config.rank_id == 0:
            return videos

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
            fps=self.pipeline.default_model_config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if "s2v" in self.pipeline.task_type:
            audio_path = "tts.wav"
            merge_video_audio(video_path=save_file, audio_path=audio_path)
