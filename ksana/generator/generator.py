from abc import ABC
import torch.distributed as dist
import ray

from ..executor import KsanaExecutor, RayKsanaExecutor
from ..utils import log, singleton
from ..utils.distribute import get_torchrun_env, is_launched_by_torchrun, get_gpu_count
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

    def __init__(self, dist_config: KsanaDistributedConfig = KsanaDistributedConfig(), offload_device="cpu"):
        """
        Initialize the KsanaGenerator.
        """
        log.info(f"Initializing KsanaGenerator with dist_config: {dist_config}, offload_device: {offload_device}")
        self.num_gpus = dist_config.num_gpus
        self._is_ray = False
        self.init_executors(dist_config=dist_config, offload_device=offload_device)

    def init_executors(self, dist_config: KsanaDistributedConfig = None, offload_device=None):
        if dist_config.num_gpus == 1:
            self.executors = KsanaExecutor(0, offload_device=offload_device)
            return
        if dist_config.num_gpus > get_gpu_count():
            raise ValueError(f"num_gpus({dist_config.num_gpus}) must be less than or equal to {get_gpu_count()}")

        if is_launched_by_torchrun():
            world_size, rank_id, local_rank_id, _ = get_torchrun_env()
            if world_size != dist_config.num_gpus:
                raise ValueError(f"world_size({world_size}) must be equal to num_gpus({dist_config.num_gpus})")
            self.executors = KsanaExecutor(device_id=local_rank_id, offload_device=offload_device)
            self.executors.init_torch_dist_group(rank_id, dist_config=dist_config)
        else:
            # ray local device id always be 0
            local_rank_id = 0
            ray.init(num_gpus=dist_config.num_gpus)
            self.executors = [
                RayKsanaExecutor.remote(local_rank_id, offload_device) for _ in range(dist_config.num_gpus)
            ]
            init_futures = []
            for rank_id, executor in enumerate(self.executors):
                future = executor.init_torch_dist_group.remote(rank_id, dist_config)
                init_futures.append(future)
            ray.get(init_futures)
            self._is_ray = True

    @staticmethod
    def from_models(
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
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
        generator = get_generator(dist_config=dist_config, offload_device=offload_device)
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
        """
        Load models from model_path.
        """
        if self.executors is None:
            raise RuntimeError("executors is not initialized")
        if self.is_ray:
            func_futures = [executor.load_models.remote(model_path, **kwargs) for executor in self.executors]
            ray.get(func_futures)
        else:
            self.executors.load_models(model_path, **kwargs)

    def load_diffusion_model(self, model_path, **kwargs):
        assert self.executors is not None, "executors is not initialized"
        if self.is_ray:
            comfy_bar_callback = kwargs.pop("comfy_bar_callback", None)
            if comfy_bar_callback is not None:
                log.info("comfy_bar_callback is not support at ray gpus, skip it")
            func_futures = [executor.load_diffusion_model.remote(model_path, **kwargs) for executor in self.executors]
            funcs_res = ray.get(func_futures)
            # note all gpus return the same model keys, so just return any result
            any_rank_id = 0
            return funcs_res[any_rank_id]
        else:
            return self.executors.load_diffusion_model(model_path, **kwargs)

    @property
    def is_ray(self):
        return self._is_ray and ray.is_initialized()

    def __del__(self):

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if ray.is_initialized():
            ray.shutdown()

    def broadcast_input_args(self, prompt, seed, prompt_negative=None):
        if dist.is_initialized():
            dist.broadcast_object_list([prompt, seed], src=0)
            if prompt_negative is not None:
                dist.broadcast_object_list(prompt_negative, src=0)

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

        if self.is_ray:
            func_futures = [
                executor.generate_video.remote(
                    prompt=prompt,
                    prompt_negative=prompt_negative,
                    sample_config=sample_config,
                    runtime_config=runtime_config,
                )
                for executor in self.executors
            ]
            res = self.get_rank0_res(ray.get(func_futures))
        else:
            rank_0_id = 0
            res = self.executors.generate_video(
                prompt=prompt,
                prompt_negative=prompt_negative,
                sample_config=sample_config,
                runtime_config=runtime_config,
            ).get(rank_0_id)

        return res

    def get_rank0_res(self, ray_res: list):
        """
        input : [{rank_id_1: [None, None, ...]}, {rank_id_0: [tensor, tensor, ...]}] or [{rank_id_1: None}, {rank_id_0: tensor}]
        return [tensor, tensor, ...] or tensor
        """
        for r in ray_res:
            if r is None:
                continue
            res = r.get(0, None)
            if res is None:
                continue
            if isinstance(res, (list, tuple)):
                if not all(res):
                    raise ValueError(f"rank 0 res has None: {res}")
            return res

    def generate_video_with_tensors(self, model, positive, negative, **kwargs):
        if self.is_ray:
            comfy_bar_callback = kwargs.pop("comfy_bar_callback", None)
            if comfy_bar_callback is not None:
                log.info("comfy_bar_callback is not support at ray gpus, skip it")
            func_futures = [
                executor.generate_video_with_tensors.remote(model=model, positive=positive, negative=negative, **kwargs)
                for executor in self.executors
            ]
            gpus_res = ray.get(func_futures)
            log.debug(f"gpus_res: {gpus_res}")
            res = self.get_rank0_res(gpus_res)
        else:
            rank_0_id = 0
            res = self.executors.generate_video_with_tensors(
                model=model, positive=positive, negative=negative, **kwargs
            ).get(rank_0_id)

        return res
