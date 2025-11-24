from abc import ABC
import torch.distributed as dist

from ..executor import KsanaExecutor
from ..utils import log, singleton

from ..utils.distribute import get_ksana_distributed_config_from_torchrun_env

from ..config import KsanaDistributedConfig


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

    def __init__(self, num_gpus: int = 1, **kwargs):
        """
        Initialize the KsanaGenerator.
        """

        log.info(f"Initializing KsanaGenerator with num_gpus: {num_gpus}, kwargs: {kwargs}")
        self.init_executors(num_gpus, **kwargs)
        self.num_gpus = num_gpus

    def init_executors(self, num_gpus: int = 1, **kwargs):
        if num_gpus > 1:
            dist_config = get_ksana_distributed_config_from_torchrun_env(**kwargs)
            if dist_config.world_size != num_gpus:
                # TODO: run with ray
                log.info(f"dist_config.world_size({dist_config.world_size}) != num_gpus({num_gpus})")
                raise ValueError(f"dist_config.world_size({dist_config.world_size}) != num_gpus({num_gpus})")
            else:
                # means using torchrun so only create one executor
                if kwargs.get("use_sp", False):
                    dist_config.use_sp = True
                self.executors = KsanaExecutor(dist_config=dist_config, **kwargs)
        else:
            self.executors = KsanaExecutor(dist_config=KsanaDistributedConfig())

    def set_executor(self, executor):
        self.executors = executor

    @classmethod
    def from_pretrained(cls, checkpoint_dir, lora_dir=None, num_gpus=1, torch_compile_config=None, **kwargs):
        """
        Load a pre-trained model.
        """
        generator = get_generator(num_gpus=num_gpus, **kwargs)
        generator.load_model_from_pretrained(
            checkpoint_dir, lora_dir=lora_dir, torch_compile_config=torch_compile_config
        )
        return generator

    def load_model_from_pretrained(self, checkpoint_dir, lora_dir=None, torch_compile_config=None, **kwargs):
        if hasattr(self.executors, "__iter__"):
            for executor in self.executors:
                executor.load_model_from_pretrained(
                    checkpoint_dir, lora_dir=lora_dir, torch_compile_config=torch_compile_config, **kwargs
                )
        else:
            self.executors.load_model_from_pretrained(
                checkpoint_dir, lora_dir=lora_dir, torch_compile_config=torch_compile_config, **kwargs
            )

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

    def generate_video(self, prompt: str, **kwargs):
        if self.num_gpus > 1:
            self.broadcast_input_args(
                prompt, kwargs.get("seed", None), prompt_negative=kwargs.get("prompt_negative", None)
            )

        if hasattr(self.executors, "__iter__"):
            # TODO: return videos
            for executor in self.executors:
                res = executor.generate_video(prompt=prompt, **kwargs)
        else:
            res = self.executors.generate_video(prompt=prompt, **kwargs)

        return res

    def generate_video_with_tensors(self, *args, **kwargs):
        if hasattr(self.executors, "__iter__"):
            for executor in self.executors:
                executor.generate_video_with_tensors(*args, **kwargs)
        else:
            self.executors.generate_video_with_tensors(*args, **kwargs)
        return self.executors
