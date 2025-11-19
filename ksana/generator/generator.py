from abc import ABC
import os
import torch
import torch.distributed as dist

from ..executor import KsanaExecutor, create_executor_config
from ..utils import log


def get_generator(*args, **kwargs):
    """
    Get the generator instance.
    """
    return KsanaGenerator(*args, **kwargs)


# TODO: singlen
# single generator
# @singleton
class KsanaGenerator(ABC):
    """
    Base class for all Ksana generators.
    """

    executors = None

    def __init__(self, *args, **kwargs):
        """
        Initialize the pipeline.
        """

        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{local_rank}")
        # _init_logging(rank)
        # log.info(f"Initializing KsanaGenerator with kwargs: {kwargs}")

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        else:
            log.info("Running in non-distributed environment.")
            # TODO: support t5 fsdp and dit fsdp
            # assert not (
            #     args.t5_fsdp or args.dit_fsdp
            # ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            # assert not (
            #     args.ulysses_size > 1
            # ), f"sequence parallel are not supported in non-distributed environments."

        # if args.ulysses_size > 1:
        #     assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        #     init_distributed_group()

        # if self.executor is None:
        #     raise ValueError("Executor must be provided.")
        # TODO: multi gpus support
        self.executors = KsanaExecutor(device=self.device, **kwargs)
        self.model = self.executors.model

        # self.executors._run_workers('initialize_wani2v', **kwargs)

    def set_executor(self, executor):
        self.executors = executor

    @classmethod
    def from_pretrained(cls, checkpoint_dir, *args, **kwargs):
        """
        Load a pre-trained model.
        """
        generator = get_generator()
        lora_dir = kwargs.get("lora_dir", None)
        executor_config = create_executor_config(checkpoint_dir, lora_dir=lora_dir)
        executor = KsanaExecutor(executor_config)
        executor.load_model(
            checkpoint_dir, lora_dir=lora_dir, torch_compile_config=kwargs.get("torch_compile_config", None)
        )
        generator.set_executor(executor)
        return generator

    def to_cpu(self):
        self.executors.to_cpu()

    def to_gpu(self):
        self.executors.to_gpu()

    # def clean(self):
    #     for worker in self.workers:
    #         ray.kill(worker)
    #     ray.util.remove_placement_group(self.placement_group)
    #     self.workers = []

    def generate_video(self, *args, **kwargs):
        return self.executors.generate_video(*args, **kwargs)

    def generate_video_with_tensors(self, *args, **kwargs):
        return self.executors.generate_video_with_tensors(*args, **kwargs)

    # def load_state_dict_from_file(self, file_path):
    #     """
    #     Load state dict from file.
    #     """
    #     return self.executor("load_state_dict_from_file", file_path)
