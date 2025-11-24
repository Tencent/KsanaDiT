from abc import ABC

import torch
import random
import sys
import os
import math

from datetime import datetime
import gc

from tqdm import tqdm
from .configs import WanExecutorConfig, KsanaExecutorConfig, WanLightLoraExecutorConfig
from .sample_schedulers import get_sample_scheduler

from ..models import KsanaModel, KsanaT5Encoder, KsanaVAE
from ..cache import create_cache, DCacheConfig
from ..utils import log, print_recursive, time_range, save_video, merge_video_audio, get_world_size
from ..utils.const import DEFAULT_OUTPUTS_VIDEO_DIR, DEFAULT_SEED
from ..config import KsanaSampleConfig, KsanaRuntimeConfig
from ..utils.profile import MemoryProfiler

# from functools import partial
# from .distributed.fsdp import shard_model


class KsanaExecutor(ABC):
    """
    Base class for all Ksana executors.
    """

    model = None

    def __init__(self, ksana_config: KsanaExecutorConfig = None, **kwargs):
        """
        Initialize the executor.
        """
        self.run_device = kwargs.get("device", torch.device("cuda"))
        self.offload_device = kwargs.get("offload_device", torch.device("cpu"))
        self.rank = 0

        # if self.executor is None:
        #     raise ValueError("Executor must be provided.")
        use_sp = kwargs.get("use_sp", False)
        dit_fsdp = kwargs.get("dit_fsdp", False)  # noqa: F841
        self.shard_fn = None  # partial(shard_model, device_id=device_id)  # noqa: F841
        convert_model_dtype = kwargs.get("convert_model_dtype", None)  # noqa: F841

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.ksana_config = ksana_config
        if not (
            isinstance(self.ksana_config, WanExecutorConfig)
            or isinstance(self.ksana_config, WanLightLoraExecutorConfig)
            or self.ksana_config is None
        ):
            raise RuntimeError(f"model_name {self.model_name} is not supported yet")

    def load_model(self, checkpoint_dir, lora_dir=None, torch_compile_config=None):
        if self.model_name != "wan2.2":
            raise RuntimeError(f"model_name {self.model_name} is not supported yet")
        if self.task_type != "t2v":
            raise RuntimeError(f"task_type {self.task_type} is not supported yet")
        # TODO: support more task type here

        self.text_encoder = KsanaT5Encoder(
            self.default_model_config, checkpoint_dir=checkpoint_dir, shard_fn=self.shard_fn
        )

        if self.task_type == "t2v":
            self.vae = KsanaVAE(
                vae_type="wan2_1",
                model_config=self.default_model_config,
                checkpoint_dir=checkpoint_dir,
                device=self.run_device,
            )

        self.high_noise_model = KsanaModel(self.default_model_config)
        self.low_noise_model = KsanaModel(self.default_model_config)
        self.high_noise_model.load(
            checkpoint_dir=checkpoint_dir,
            subfolder=self.default_model_config.high_noise_checkpoint,
            lora_dir=(
                os.path.join(lora_dir, self.ksana_config.high_noise_lora_checkpoint) if lora_dir is not None else None
            ),
            torch_compile_config=torch_compile_config,
        )
        self.low_noise_model.load(
            checkpoint_dir=checkpoint_dir,
            subfolder=self.default_model_config.low_noise_checkpoint,
            lora_dir=(
                os.path.join(lora_dir, self.ksana_config.low_noise_lora_checkpoint) if lora_dir is not None else None
            ),
            torch_compile_config=torch_compile_config,
        )

    def process_input_sample_config(self, **kwargs):
        default_model_config = self.ksana_config.default_model_config

        denoise = kwargs.get("denoise", 1.0)
        sampling_steps = kwargs.get("steps", self.ksana_config.steps)
        cfg_scale = kwargs.get("cfg_scale", self.ksana_config.cfg_scale)
        sample_shift = kwargs.get("sample_shift", self.ksana_config.sample_shift)
        sample_solver = kwargs.get("sample_solver", self.ksana_config.sample_solver)
        if sampling_steps is None:
            sampling_steps = default_model_config.get("sample_steps", None)
        if sample_shift is None:
            sample_shift = default_model_config.get("sample_shift", None)
        if cfg_scale is None:
            cfg_scale = default_model_config.get("sample_guide_scale", None)
        if sample_solver is None:
            sample_solver = default_model_config.get("sample_solver", None)

        sample_config = KsanaSampleConfig(
            steps=sampling_steps,
            cfg_scale=cfg_scale,
            shift=sample_shift,
            solver=sample_solver,
            denoise=denoise,
        )
        return sample_config

    def process_input_runtime_config(self, **kwargs):
        default_model_config = self.ksana_config.default_model_config
        input_size = kwargs.get("size", default_model_config.get("size", None))
        input_frame_num = kwargs.get("frame_num", default_model_config.get("frame_num", None))
        seed = kwargs.get("seed", DEFAULT_SEED)
        boundary = kwargs.get("boundary", default_model_config.boundary)
        # TODO: input dtype should be str as float, float16, float32, give some map or just use torch.dtype
        run_dtype = kwargs.get("run_dtype", None)
        if run_dtype is None:
            run_dtype = default_model_config.get("param_dtype", None)

        runtime_config = KsanaRuntimeConfig(
            input_size=input_size,
            input_frame_num=input_frame_num,
            seed=seed,
            run_dtype=run_dtype,
            boundary=boundary,
        )
        return runtime_config

    def process_input_cache(self, cache_method):
        high_cache_config = None
        low_cache_config = None
        if cache_method == "DCache":
            high_cache_config = DCacheConfig(
                fast_degree=70,
                slow_degree=35,
                fast_force_calc_every_n_steps=1,
                slow_force_calc_every_n_steps=5,
                name="high_dcache",
            )
            low_cache_config = DCacheConfig(
                fast_degree=65,
                slow_degree=25,
                fast_force_calc_every_n_steps=2,
                slow_force_calc_every_n_steps=4,
                name="low_dcache",
            )
        return high_cache_config, low_cache_config

    @time_range
    def generate_video(self, prompt, **kwargs):
        """_summary_

        Args:
            prompt (_type_): _description_
            steps (int, optional): _description_.
            size (tuple, optional): _description_. Defaults to (1024, 720).
            frame_num (int, optional): _description_. Defaults to 81.
            sample_shift (float, optional): _description_.
            cache_method (str, optional): _description_.
        Returns:
            _type_: _description_
        """
        default_model_config = self.ksana_config.default_model_config
        prompt_positive = kwargs.get("prompt", None) if prompt is None else prompt
        prompt_negative = kwargs.get("prompt_negative", default_model_config.sample_neg_prompt)
        # TODO: offload_model should be as a config
        offload_model = True

        # TODO: maybe batch prompt for text encoder
        if prompt_positive is not None:
            positive = self.text_encoder.forward([prompt_positive])[0]
        if prompt_negative is not None:
            negative = self.text_encoder.forward([prompt_negative])[0]
        if offload_model:
            self.text_encoder.to(self.offload_device)

        positive = positive.unsqueeze(0)
        negative = negative.unsqueeze(0)
        sample_config = self.process_input_sample_config(**kwargs)
        runtime_config = self.process_input_runtime_config(**kwargs)
        high_cache_config, low_cache_config = self.process_input_cache(kwargs.get("cache_method", None))

        latents = self.generate_video_with_tensors(
            size=kwargs.get("size", default_model_config.size),
            frame_num=kwargs.get("frame_num", default_model_config.frame_num),
            high_model=self.high_noise_model,
            low_model=self.low_noise_model,
            positive=positive,
            negative=negative,
            latents=None,
            sample_config=sample_config,
            runtime_config=runtime_config,
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
        )
        if offload_model:
            self.high_noise_model.to(self.offload_device)
            self.low_noise_model.to(self.offload_device)
        del positive, negative

        if self.rank == 0:
            videos = self.vae.decode(latents)[0]
            del latents
            log.info(f"Generated video shape: {videos.shape}")

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        output_folder = kwargs.get("output_folder", DEFAULT_OUTPUTS_VIDEO_DIR)
        return_frames = kwargs.get("return_frames", False)
        self.save_video(videos, self.get_save_path(output_folder, prompt_positive))

        # if dist.is_initialized():
        #     dist.barrier()
        #     dist.destroy_process_group()
        if not return_frames:
            del videos
        return videos

    def get_save_path(self, output_folder, prompt_positive):
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt_positive.replace(" ", "_").replace("/", "_")[:30]
        suffix = ".mp4"
        save_file = (
            f"{self.ksana_config.default_model_config.model_name}_{self.ksana_config.default_model_config.task_type}_"
            + f"{self.ksana_config.default_model_config.model_size}_{formatted_time}_{formatted_prompt}"
            + suffix
        )
        return os.path.join(output_folder, save_file)

    def save_video(self, video, save_file):
        if self.rank != 0:
            return
        log.info(f"Saving generated video to {save_file}")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=self.ksana_config.default_model_config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        if "s2v" in self.ksana_config.default_model_config.task_type:
            audio_path = "tts.wav"
            merge_video_audio(video_path=save_file, audio_path=audio_path)

    def set_cache(self, model_type: str, cache_config):
        self.cache = cache_config

    @time_range
    def generate_video_with_tensors(
        self,
        high_model: KsanaModel,
        positive: torch.Tensor,  # [bs, 512, 4096]
        negative: torch.Tensor,  # [bs, 512, 4096]
        latents: torch.Tensor,  # [bs, 16, 2, 90, 160]
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        low_model: KsanaModel = None,
        high_cache_config=None,
        low_cache_config=None,
        *args,
        **kwargs,
    ):
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_config (KsanaSampleConfig): _description_
            cache (_type_, optional): _description_. Defaults to None.

        Returns:
            latents (torch.Tensor)
        """
        log.info(f"runtime_config: {runtime_config}, sample_config: {sample_config}")
        assert sample_config.denoise == 1.0, f"only support denoise ==1.0 yet, but got {sample_config.denoise}"
        assert sample_config.solver in [
            "uni_pc",
            "dpm++",
            "euler",
        ], f"sample_solver {sample_config.solver} not supported in list: ['uni_pc', 'dpm++', 'euler']"
        log.debug("latents, positive, negtive:")
        print_recursive(latents, log.debug)
        print_recursive(positive, log.debug)
        print_recursive(negative, log.debug)

        if isinstance(sample_config.cfg_scale, float):
            high_sample_guide_scale = sample_config.cfg_scale
            low_sample_guide_scale = sample_config.cfg_scale
        elif hasattr(sample_config.cfg_scale, "__len__") and len(sample_config.cfg_scale) >= 2:
            low_sample_guide_scale = sample_config.cfg_scale[0]
            high_sample_guide_scale = sample_config.cfg_scale[1]
        else:
            raise ValueError(f"sample_config.cfg_scale {sample_config.cfg_scale} not supported")
        cfg_scale = high_sample_guide_scale

        if low_model is not None:
            boundary = (
                runtime_config.boundary if runtime_config.boundary is not None else low_model.default_config.boundary
            )
        else:
            low_sample_guide_scale = None
            boundary = None

        low_cache = None
        high_cache = None
        if high_cache_config is not None:
            high_cache = create_cache(
                f"{high_model.model_name}-high",
                high_model.task_type,
                high_model.model_size,
                high_cache_config,
            )
        if low_cache_config is not None:
            low_cache = create_cache(
                f"{low_model.model_name}-low",
                low_model.task_type,
                low_model.model_size,
                low_cache_config,
            )
        seed = runtime_config.seed if runtime_config.seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.run_device)
        seed_g.manual_seed(seed)

        if latents is None:
            size = kwargs.get("size", (1280, 720))
            frame_num = kwargs.get("frame_num", 81)
            target_shape = (
                self.vae.z_dim,
                (frame_num - 1) // self.vae.vae_stride[0] + 1,
                size[1] // self.vae.vae_stride[1],
                size[0] // self.vae.vae_stride[2],
            )
            # (z_dim:16, 2, 96, 160)
        else:
            if len(latents.shape) == 5:
                latents = latents.squeeze(0)
            target_shape = latents.shape
        # [bs, input_text_len, 4096]
        assert positive.ndim == negative.ndim == 3, f"positive.shape {positive.shape}, negative.shape {negative.shape}"

        # used to target_shape, self.patch_size, sp_size
        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (high_model.default_model_config.patch_size[1] * high_model.default_model_config.patch_size[2])
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
        # TODO: support bs > 1
        latents = latents.unsqueeze(0)

        # # model
        # @contextmanager
        # def noop_no_sync():
        #     yield
        # no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
        #                             noop_no_sync)
        # no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
        #                              noop_no_sync)
        run_dtype = runtime_config.run_dtype if runtime_config.run_dtype is not None else high_model.run_dtype
        latents = self.cast_to(latents, run_dtype, self.run_device)

        with (
            # torch.amp.autocast("cuda", dtype=run_dtype), # 应该不需要
            torch.no_grad(),
            # no_sync(),# 作用是什么
        ):
            positive = self.cast_to(positive, run_dtype, self.run_device)
            negative = self.cast_to(negative, run_dtype, self.run_device)
            if boundary is not None:
                boundary = boundary * low_model.default_model_config.num_train_timesteps
            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=high_model.default_model_config.num_train_timesteps,
                sampling_steps=sample_config.steps,
                sample_solver=sample_config.solver,
                device=self.run_device,
                shift=sample_config.shift,
            )
            arg_c = {"phase": "cond", "context": positive, "seq_len": seq_len}
            arg_null = {"phase": "uncond", "context": negative, "seq_len": seq_len}
            log.debug(f"timesteps: {timesteps}, boundary:{boundary}, seq_len:{seq_len}")
            # timesteps: tensor([999, 997, ...])
            MemoryProfiler.record_memory("before_inference_loop")

            for iter_id, t in enumerate(tqdm(timesteps)):
                MemoryProfiler.record_memory(f"before_inference_loop_iter_{iter_id}")
                # [bs, 16, fi, hi, wi]
                latent_model_input = latents
                # t : tensor(999)
                if low_model is not None and boundary is not None and t.item() < boundary:
                    cfg_scale = low_sample_guide_scale
                timestep = [t]
                timestep = torch.stack(timestep)  # [tensor] => tensor([])
                timestep_id = t.item()
                run_model = self.get_run_model(
                    high_model=high_model,
                    low_model=low_model,
                    timestep_id=timestep_id,
                    boundary=boundary,
                )
                run_model = run_model.to(self.run_device)
                MemoryProfiler.record_memory(f"inference_step_{iter_id}_after_model_switch")
                run_cache = self.get_run_cache(
                    high_cache=high_cache,
                    low_cache=low_cache,
                    timestep_id=timestep_id,
                    boundary=boundary,
                )
                # TODO: concat cond and uncond context to forward once
                # arg_c["context"] = [torch.cat([positive, negative], dim=0)]

                # [bs, 16, fi, hi, wi] => [bs, 16, fi, hi, wi]
                noise_pred_cond = run_model.forward(x=latent_model_input, t=timestep, cache=run_cache, **arg_c)
                if self.use_cfg(cfg_scale):
                    noise_pred_uncond = run_model.forward(x=latent_model_input, t=timestep, cache=run_cache, **arg_null)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # [bs, 16, fi, hi, wi] => [bs, 16, fi, hi, wi]
                temp_x0 = sample_scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    generator=seed_g,
                )
                latents = temp_x0 if sample_config.solver == "euler" else temp_x0[0]
                MemoryProfiler.record_memory(f"inference_step_{iter_id}_after_sample_scheduler")

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
        log.debug(f"latents shape: {latents.shape}")
        # [bs, 16, fi, hi, wi]
        return latents

    def use_cfg(self, cfg_scale: float, eps: float = 1e-6):
        return abs(cfg_scale - 1.0) > eps

    def cast_to(self, src, dtype: torch.dtype, device: torch.device):
        if src.dtype != dtype:
            # TODO: add more check for dtype, only allow
            src = src.to(dtype)
        if src.device != device:
            src = src.to(device)
        return src

    def get_run_model(self, high_model, low_model, timestep_id: int, boundary: float):
        if low_model is None:
            return high_model
        assert boundary is not None, "boundary must be provided when low_model is not None"
        if timestep_id >= boundary:
            if low_model.device != self.offload_device:
                low_model.to(self.offload_device)
            return high_model
        else:
            if high_model.device != self.offload_device:
                high_model.to(self.offload_device)
            return low_model

    def get_run_cache(self, high_cache, low_cache, timestep_id, boundary):
        if low_cache is None:
            return high_cache
        if timestep_id >= boundary:
            return high_cache
        else:
            return low_cache

    @property
    def model_name(self):
        return self.default_model_config.model_name

    @property
    def task_type(self):
        return self.default_model_config.task_type

    @property
    def model_size(self):
        return self.default_model_config.model_size

    @property
    def default_model_config(self):
        return self.ksana_config.default_model_config

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
