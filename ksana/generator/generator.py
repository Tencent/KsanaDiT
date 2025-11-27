from abc import ABC
import torch.distributed as dist

from ..executor import KsanaExecutor
from ..utils import log, singleton

from ..utils.distribute import get_ksana_distributed_config_from_torchrun_env

from ..config import KsanaDistributedConfig, KsanaSampleConfig, KsanaRuntimeConfig, KsanaModelConfig


def get_generator(*args, **kwargs):
    """
    Get the generator instance.
    """
    return KsanaGenerator(*args, **kwargs)


@singleton
class KsanaGenerator(ABC):
    """
    Base class for all Ksana generators.
    """

    executors = None

    def __init__(self, num_gpus: int = 1, dist_config=None, offload_device="cpu"):
        """
        Initialize the KsanaGenerator.
        """

        log.info(
            f"Initializing KsanaGenerator with num_gpus: {num_gpus}, dist_config: {dist_config}, offload_device: {offload_device}"
        )
        self.init_executors(num_gpus, dist_config=dist_config, offload_device=offload_device)
        self.num_gpus = num_gpus

    def init_executors(self, num_gpus: int = 1, dist_config=None, offload_device=None):
        if num_gpus > 1:
            dist_config = get_ksana_distributed_config_from_torchrun_env(dist_config)
            if dist_config.world_size != num_gpus:
                # TODO: run with ray
                log.info(f"dist_config.world_size({dist_config.world_size}) != num_gpus({num_gpus})")
                raise ValueError(f"dist_config.world_size({dist_config.world_size}) != num_gpus({num_gpus})")
            else:
                # means using torchrun so only create one executor
                self.executors = KsanaExecutor(dist_config=dist_config, offload_device=offload_device)
        else:
            self.executors = KsanaExecutor(dist_config=KsanaDistributedConfig())

    @staticmethod
    def from_models(
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        num_gpus: int = 1,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        dist_config: KsanaDistributedConfig = None,
        offload_device="cpu",
        **kwargs,
    ):
        """
        Load a pre-trained model.
        """
        model_config = model_config or KsanaModelConfig()
        dist_config = dist_config or KsanaDistributedConfig()
        generator = get_generator(num_gpus=num_gpus, dist_config=dist_config, offload_device=offload_device)
        generator.load_models(
            model_path,
            text_checkpoint_dir=text_checkpoint_dir,
            vae_checkpoint_dir=vae_checkpoint_dir,
            lora_dir=lora_dir,
            model_config=model_config,
            **kwargs,
        )
        return generator

    def load_models(self, model_path, **kwargs):
        assert self.executors is not None, "executors is not initialized"
        if hasattr(self.executors, "__iter__"):
            for executor in self.executors:
                executor.load_models(model_path, **kwargs)
        else:
            self.executors.load_models(model_path, **kwargs)

    def load_diffusion_model(self, model_path, **kwargs):
        assert self.executors is not None, "executors is not initialized"
        if hasattr(self.executors, "__iter__"):
            # TODO: how to return multiple ksana_model
            res = []
            for executor in self.executors:
                res.append(executor.load_diffusion_model(model_path, **kwargs))
        else:
            res = self.executors.load_diffusion_model(model_path, **kwargs)
        return res

    # def clean(self):
    #     for worker in self.workers:
    #         ray.kill(worker)
    #     ray.util.remove_placement_group(self.placement_group)
    #     self.workers = []

    def broadcast_input_args(self, prompt, seed, prompt_negative=None):
        if dist.is_initialized():
            dist.broadcast_object_list([prompt, seed], src=0)
            if prompt_negative is not None:
                dist.broadcast_object_list(prompt_negative, src=0)
        # TODO: ray way to broadcast prompt

    def generate_video(
        self,
        prompt: str,
        prompt_negative: str = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        **kwargs,
    ):
        if len(kwargs) > 0:
            log.warning(f"kwargs {kwargs} are not used")
        if self.num_gpus > 1:
            self.broadcast_input_args(
                prompt, runtime_config.seed if runtime_config else None, prompt_negative=prompt_negative
            )

        if hasattr(self.executors, "__iter__"):
            # TODO: return videos
            for executor in self.executors:
                res = executor.generate_video(
                    prompt=prompt,
                    prompt_negative=prompt_negative,
                    sample_config=sample_config,
                    runtime_config=runtime_config,
                )
        else:
            res = self.executors.generate_video(
                prompt=prompt,
                prompt_negative=prompt_negative,
                sample_config=sample_config,
                runtime_config=runtime_config,
            )

        return res

    def generate_video_with_tensors(self, model, positive, negative, **kwargs):
        if hasattr(self.executors, "__iter__"):
            res = []
            for executor in self.executors:
                res.append(
                    executor.generate_video_with_tensors(model=model, positive=positive, negative=negative, **kwargs)
                )
        else:
            res = self.executors.generate_video_with_tensors(
                model=model, positive=positive, negative=negative, **kwargs
            )
        return res
