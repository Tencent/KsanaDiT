from abc import ABC

import torch
import random
import sys
import math

from tqdm import tqdm

from .sample_schedulers import get_sample_scheduler

from ..models import vDitModel
from ..cache import create_cache
from ..distributed.utils import get_world_size
from ..utils import log, print_recursive, time_range

# from functools import partial
# from .distributed.fsdp import shard_model


class vDitExecutor(ABC):
    """
    Base class for all vDit executors.
    """

    model = None

    def __init__(self, **kwargs):
        """
        Initialize the executor.
        """
        self.run_device = kwargs.get("device", torch.device("cuda"))
        self.offload_device = kwargs.get("offload_device", torch.device("cpu"))

        # if self.executor is None:
        #     raise ValueError("Executor must be provided.")
        use_sp = kwargs.get("use_sp", False)
        dit_fsdp = kwargs.get("dit_fsdp", False)  # noqa: F841
        shard_fn = None  # partial(shard_model, device_id=device_id)  # noqa: F841
        convert_model_dtype = kwargs.get("convert_model_dtype", None)  # noqa: F841

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

    def set_cache(self, model_type: str, cache_config):
        self.cache = cache_config

    @time_range
    def run(
        self,
        high_model: vDitModel,
        positive: torch.Tensor,  # [bs, 512, 4096]
        negative: torch.Tensor,  # [bs, 512, 4096]
        latents: torch.Tensor,  # [bs, 16, 2, 90, 160]
        seed: int,
        sample_shift: float,
        sample_guide_scale: float,
        denoise: float,
        sampling_steps: int = 50,
        sample_solver: str = "uni_pc",
        low_model: vDitModel = None,
        boundary: float = None,
        low_sample_guide_scale: float = None,
        high_cache_config=None,
        low_cache_config=None,
        #              size=(1280, 720),
        #              frame_num=81,
        #              offload_model=True,
        #              cache_args=None):
        # TODO: add run dtype
        # run_dtype = None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_shift (float): _description_
            sample_guide_scale (float): _description_
            denoise (float): _description_
            sampling_steps (int, optional): _description_. Defaults to 50.
            sample_solver (str, optional): _description_. Defaults to "uni_pc".
            cache (_type_, optional): _description_. Defaults to None.

        Returns:
            latents (torch.Tensor)
        """
        log.info(
            f"seed: {seed}, sample_shift: {sample_shift}, sample_guide_scale: {sample_guide_scale}, denoise: {denoise}, "
            f"sampling_steps: {sampling_steps}, sample_solver: {sample_solver}, boundary: {boundary}, low_sample_guide_scale: {low_sample_guide_scale}"
        )
        assert sample_solver in [
            "uni_pc",
            "dpm++",
        ], f"sample_solver {sample_solver} not supported in list: ['uni_pc', 'dpm++']"
        log.info("latents, positive, negtive:")
        print_recursive(latents)
        print_recursive(positive)
        print_recursive(negative)
        high_cache = create_cache(
            f"{high_model.model_kind}-high",
            high_model.task_type,
            high_model.model_size,
            high_cache_config,
        )
        if low_model is not None:
            low_cache = create_cache(
                f"{low_model.model_kind}-low",
                low_model.task_type,
                low_model.model_size,
                low_cache_config,
            )
            low_sample_guide_scale = (
                low_sample_guide_scale
                if low_sample_guide_scale is not None
                else low_model.default_config.sample_guide_scale[0]
            )
            boundary = boundary if boundary is not None else low_model.default_config.boundary
        else:
            low_sample_guide_scale = None
            boundary = None

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.run_device)
        seed_g.manual_seed(seed)

        if len(latents.shape) == 5:
            latents = latents.squeeze(0)
        if len(positive.shape) == 3:
            positive = positive.squeeze(0)
        if len(negative.shape) == 3:
            negative = negative.squeeze(0)

        target_shape = latents.shape

        # used to target_shape, self.patch_size, sp_size
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (high_model.default_config.patch_size[1] * high_model.default_config.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )  # 7200

        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.run_device,
            generator=seed_g,
        )

        latents = noise  # maybe added input latents

        # # model
        # @contextmanager
        # def noop_no_sync():
        #     yield
        # no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
        #                             noop_no_sync)
        # no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
        #                              noop_no_sync)
        run_dtype = high_model.default_config.param_dtype
        latents = self.cast_to(latents, run_dtype, self.run_device)

        with (
            torch.amp.autocast("cuda", dtype=run_dtype),
            torch.no_grad(),
            # no_sync(),# 作用是什么
        ):
            positive = self.cast_to(positive, run_dtype, self.run_device)
            negative = self.cast_to(negative, run_dtype, self.run_device)
            if boundary is not None:
                boundary = boundary * low_model.default_config.num_train_timesteps
            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=high_model.default_config.num_train_timesteps,
                sampling_steps=sampling_steps,
                sample_solver=sample_solver,
                device=self.run_device,
                shift=sample_shift,
            )

            arg_c = {
                "phase": "cond",
                "context": [positive],
                "seq_len": seq_len,
            }  # , seq_len:720, 与frame_num相关
            arg_null = {"phase": "uncond", "context": [negative], "seq_len": seq_len}
            log.info(f"timesteps: {timesteps}")
            latents = [latents]
            # timesteps: tensor([999, 997, ...])
            for iter_id, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                # t : tensor(999)
                if boundary is not None and low_model is not None:
                    assert low_sample_guide_scale is not None
                    sample_guide_scale = sample_guide_scale if t.item() >= boundary else low_sample_guide_scale
                timestep = [t]
                timestep = torch.stack(timestep)  # [tensor] => tensor([])
                timestep_id = t.item()
                run_model = self.get_run_model(
                    high_model=high_model,
                    low_model=low_model,
                    timestep_id=timestep_id,
                    boundary=boundary,
                )
                run_model = self.cast_to(run_model, run_dtype, self.run_device)
                run_cache = self.get_run_cache(
                    high_cache=high_cache,
                    low_cache=low_cache,
                    timestep_id=timestep_id,
                    boundary=boundary,
                )

                noise_pred_cond = run_model.forward(latent_model_input, t=timestep, cache=run_cache, **arg_c)[0]
                noise_pred_uncond = run_model.forward(latent_model_input, t=timestep, cache=run_cache, **arg_null)[0]

                noise_pred = noise_pred_uncond + sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[
                    0
                ]  # [1, 16, 2, 90, 160]
                latents = [temp_x0.squeeze(0)]
            if high_cache is not None:
                high_cache.show_cache_rate()
            if low_cache is not None:
                low_cache.show_cache_rate()

        del sample_scheduler

        # self.model.to(self.offload_device)

        # if offload_model:
        #     gc.collect()
        #     torch.cuda.synchronize()
        # if dist.is_initialized():
        #     dist.barrier()
        log.info(f"latents shape: {latents[0].shape}")
        return latents[0]

    def cast_to(self, src, dtype: torch.dtype, device: torch.device):
        if src.dtype != dtype:
            src = src.to(dtype)
        if src.device != device:
            src = src.to(device)
        return src

    def get_run_model(self, high_model, low_model, timestep_id: int, boundary: float):
        if low_model is None:
            return high_model
        assert boundary is not None, "boundary must be provided when low_model is not None"
        if timestep_id <= boundary:
            if high_model.device != self.offload_device:
                high_model.to(self.offload_device)
            return low_model
        else:
            if low_model.device != self.offload_device:
                low_model.to(self.offload_device)
            return high_model

    def get_run_cache(self, high_cache, low_cache, timestep_id, boundary):
        if low_cache is None:
            return high_cache
        if timestep_id <= boundary:
            return low_cache
        else:
            return high_cache

    # def prepare_cache(self, cache_args: dict =None):
    #     """
    #     Prepare cache for sampling.
    #     """
    #     if cache_args is None:
    #         cache_args = {}
    #     timesteps_range = [start, end]
    #     cache_params = vCacheParams(cache_args)
    #     high_degree, low_degree
    #     self.cache = vCache(cache_params)
