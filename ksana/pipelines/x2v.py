from abc import ABC, abstractmethod
import torch

from dataclasses import dataclass, field
from ..config import KsanaSampleConfig, KsanaRuntimeConfig, KsanaPipelineConfig, KsanaModelConfig
from ..utils import log, print_recursive, time_range, MemoryProfiler, is_dir
from ..cache import create_cache
import random
import sys
from ..sample_solvers import get_sample_scheduler
import math
from ..models import KsanaDiffusionModel, KsanaT5Encoder, KsanaVAE
from tqdm import tqdm

from typing import Optional


@dataclass(frozen=True)
class KsanaDefaultArgs:
    """
    Base configuration class for Ksana executors.
    """

    steps: int = field(default=50)
    cfg_scale: float | tuple[float, float] = field(default=None)
    sample_shift: float = field(default=None)
    sample_solver: str | None = field(default=None)


class KsanaX2VPipeline(ABC):
    def __init__(self, pipeline_config: KsanaPipelineConfig):
        """_summary_

        Args:
            pipeline_config (_type_): _description_
        """
        self.pipeline_config = pipeline_config
        self.default_args = KsanaDefaultArgs()

        self.text_encoder = None
        self.diffusion_model = None
        self.vae = None
        self.has_lora = False

    @abstractmethod
    def load_text_encoder(self, checkpoint_dir, shard_fn=None) -> KsanaT5Encoder:
        pass

    @abstractmethod
    def load_vae(self, checkpoint_dir, device) -> KsanaVAE:
        pass

    @abstractmethod
    def load_diffusion_model(
        self,
        model_path,
        *,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        comfy_model_config=None,
        comfy_model_state_dict=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ) -> KsanaDiffusionModel | tuple[KsanaDiffusionModel, KsanaDiffusionModel]:
        pass

    def offload_diffusion_model_to(self, offload_device):
        if hasattr(self.diffusion_model, "__iter__"):
            for module in self.diffusion_model:
                module.to(offload_device)
        else:
            self.diffusion_model.to(offload_device)

    def load_models(
        self,
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ) -> Optional[KsanaDiffusionModel]:
        if not is_dir(model_path):
            assert (
                text_checkpoint_dir is not None
            ), f"text_checkpoint_dir must be provided when loading from local checkpoint with diffusion model {model_path}"
            assert (
                vae_checkpoint_dir is not None
            ), f"vae_checkpoint_dir must be provided when loading from local checkpoint with diffusion model {model_path}"
        else:
            text_checkpoint_dir = model_path
            vae_checkpoint_dir = model_path
        # keep lora flag for output name
        self.has_lora = lora_dir is not None

        self.text_encoder = self.load_text_encoder(text_checkpoint_dir, shard_fn=shard_fn)
        if offload_device:
            self.text_encoder.to(offload_device)
        self.diffusion_model = self.load_diffusion_model(
            model_path,
            lora_dir=lora_dir,
            model_config=model_config,
            dist_config=dist_config,
            device=device,
            offload_device=offload_device,
            shard_fn=shard_fn,
        )
        if offload_device is not None:
            self.offload_diffusion_model_to(offload_device)
        self.vae = self.load_vae(vae_checkpoint_dir, device)
        if offload_device is not None:
            self.vae.to(offload_device)

    def forward_text_encoder(self, prompt, prompt_negative=None, device=None, offload_device=None):
        default_pipeline_config = self.pipeline_config.default_config
        prompt_positive = prompt
        prompt_negative = prompt_negative if prompt_negative is not None else default_pipeline_config.sample_neg_prompt

        assert device is not None

        if self.text_encoder.device != device:
            self.text_encoder.to(device)
        # TODO: maybe batch prompt for text encoder
        positive = self.text_encoder.forward([prompt_positive])[0]
        # [1, text_tokens, dim]
        positive = positive.unsqueeze(0)
        negative = None
        if prompt_negative is not None:
            negative = self.text_encoder.forward([prompt_negative])[0]
            negative = negative.unsqueeze(0)
        if offload_device is not None and offload_device != device:
            self.text_encoder.to(offload_device)

        return positive, negative

    @abstractmethod
    def process_input_cache(self, cache_method):
        pass

    @abstractmethod
    def forward_diffusion_model(self, positive, negative, device=None, offload_device=None, **kwargs):
        pass

    def forward_vae(self, latents, local_rank, device=None, offload_device=None):
        # TODO: support multi gpu
        if local_rank != 0:
            return
        if self.vae.device != device:
            self.vae.to(device)
        videos = self.vae.decode(latents)[0]
        if offload_device is not None and offload_device != device:
            self.vae.to(offload_device)
        del latents
        log.info(f"Generated video shape: {videos.shape}")
        return videos

    def use_cfg(self, cfg_scale: float, eps: float = 1e-6):
        return abs(cfg_scale - 1.0) > eps

    def cast_to(self, src, dtype: torch.dtype, device: torch.device):
        if src.dtype != dtype:
            src = src.to(dtype)
        if src.device != device:
            src = src.to(device)
        return src

    def get_run_model(self, high_model, low_model, timestep_id: int, boundary: float, offload_device=None):
        if low_model is None:
            return high_model
        assert boundary is not None, "boundary must be provided when low_model is not None"
        if timestep_id >= boundary:
            if low_model.device != offload_device:
                low_model.to(offload_device)
            return high_model
        else:
            if high_model.device != offload_device:
                high_model.to(offload_device)
            return low_model

    def get_run_cache(self, high_cache, low_cache, timestep_id, boundary):
        if low_cache is None:
            return high_cache
        if timestep_id >= boundary:
            return high_cache
        else:
            return low_cache

    def valid_args(self, model, sample_config):
        high_model = model
        low_model = None
        if hasattr(model, "__len__"):
            assert len(model) == 2, f"size of model must be 2, but got {len(model)}"
            high_model, low_model = model
        if isinstance(sample_config.cfg_scale, float):
            high_sample_guide_scale = sample_config.cfg_scale
            low_sample_guide_scale = None if low_model is None else sample_config.cfg_scale
        elif hasattr(sample_config.cfg_scale, "__len__"):
            assert (
                len(sample_config.cfg_scale) == 2
            ), f"size of cfg_scale must be 2, but got {len(sample_config.cfg_scale)}"
            low_sample_guide_scale = sample_config.cfg_scale[0]
            high_sample_guide_scale = sample_config.cfg_scale[1]
        else:
            raise ValueError(f"sample_config.cfg_scale {sample_config.cfg_scale} not supported")
        assert sample_config.denoise == 1.0, f"only support denoise ==1.0 yet, but got {sample_config.denoise}"
        return high_model, low_model, high_sample_guide_scale, low_sample_guide_scale

    def prepare_sample_default_args(self, sample_config: KsanaSampleConfig):
        # input sample_config > KsanaDefaultArgs > default_pipeline_config
        default_pipeline_config = self.pipeline_config.default_config
        sample_default_args = {
            "steps": (
                self.default_args.steps
                if self.default_args.steps is not None
                else default_pipeline_config.get("sample_steps", None)
            ),
            "cfg_scale": (
                self.default_args.cfg_scale
                if self.default_args.cfg_scale is not None
                else default_pipeline_config.get("sample_guide_scale", None)
            ),
            "sample_shift": (
                self.default_args.sample_shift
                if self.default_args.sample_shift is not None
                else default_pipeline_config.get("sample_shift", None)
            ),
            "sample_solver": (
                self.default_args.sample_solver
                if self.default_args.sample_solver is not None
                else default_pipeline_config.get("sample_solver", None)
            ),
            "denoise": default_pipeline_config.get("denoise", None),
        }
        return KsanaSampleConfig.copy_with_default(sample_config, sample_default_args)

    def create_random_noise_latents(
        self, latents, runtime_config: KsanaRuntimeConfig, device: torch.device, dtype: torch.dtype
    ):
        """_summary_

        Args:
            return tensor shape :[bs, z_dim, f, h, w]
        """
        seed = (
            runtime_config.seed
            if runtime_config.seed is not None and runtime_config.seed >= 0
            else random.randint(0, sys.maxsize)
        )
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        bs = 1
        if latents is None:
            size = runtime_config.size
            frame_num = runtime_config.frame_num
            target_shape = (
                self.vae.z_dim,
                (frame_num - 1) // self.vae.vae_stride[0] + 1,
                size[1] // self.vae.vae_stride[1],
                size[0] // self.vae.vae_stride[2],
            )
            # => (z_dim:16, 2, 96, 160)
        else:
            if len(latents.shape) == 5:
                bs = latents.shape[0]
                target_shape = latents.squeeze(0).shape

        noise = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g,
        ).to(dtype)
        if bs > 1:
            noise = noise.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        else:
            noise = noise.unsqueeze(0)
        return noise, seed_g

    def get_seq_len(self, target_shape, patch_size: list[int], sp_size: int):
        _, _, f, h, w = target_shape
        return math.ceil((h * w) / (patch_size[1] * patch_size[2]) * f / sp_size) * sp_size

    @time_range
    def generate_video_with_tensors(
        self,
        model: KsanaDiffusionModel | tuple[KsanaDiffusionModel, KsanaDiffusionModel],
        positive: torch.Tensor,  # [bs, 512, 4096]
        negative: torch.Tensor,  # [bs, 512, 4096]
        latents: torch.Tensor,  # [bs, 16, 2, 90, 160]
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        high_cache_config=None,
        low_cache_config=None,
        device=None,
        offload_device=None,
    ):
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_config (KsanaSampleConfig): _description_
        Returns:
            latents (torch.Tensor)
        """
        log.info(f"runtime_config: {runtime_config}, sample_config: {sample_config}")
        high_model, low_model, high_sample_guide_scale, low_sample_guide_scale = self.valid_args(model, sample_config)
        log.info(
            f"high_sample_guide_scale: {high_sample_guide_scale}, low_sample_guide_scale: {low_sample_guide_scale}"
        )
        log.debug("latents, positive, negtive:")
        assert (
            low_model is None or high_model.run_dtype == low_model.run_dtype
        ), f"high_model.run_dtype {high_model.run_dtype}, low_model.run_dtype {low_model.run_dtype} should be same"
        run_dtype = high_model.run_dtype
        print_recursive(latents, log.debug)
        print_recursive(positive, log.debug)
        print_recursive(negative, log.debug)
        # [bs, input_text_len, 4096]
        assert positive.ndim == negative.ndim == 3, f"positive.shape {positive.shape}, negative.shape {negative.shape}"

        default_pipeline_config = self.pipeline_config.default_config
        boundary = None if low_model is None else runtime_config.boundary * default_pipeline_config.num_train_timesteps

        low_cache = None
        high_cache = None
        if high_cache_config is not None:
            high_cache = create_cache(
                f"{high_model.model_name}-high",
                high_model.task_type,
                high_model.model_size,
                high_cache_config,
            )
        if low_cache_config is not None and low_model is not None:
            low_cache = create_cache(
                f"{low_model.model_name}-low",
                low_model.task_type,
                low_model.model_size,
                low_cache_config,
            )

        latents, seed_g = self.create_random_noise_latents(latents, runtime_config, device, run_dtype)
        seq_len = self.get_seq_len(latents.shape, default_pipeline_config.patch_size, high_model.sp_size)

        latents = self.cast_to(latents, run_dtype, device)
        positive = self.cast_to(positive, run_dtype, device)
        negative = self.cast_to(negative, run_dtype, device)
        with torch.no_grad():
            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=default_pipeline_config.num_train_timesteps,
                sampling_steps=sample_config.steps,
                sample_solver=sample_config.solver,
                device=device,
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
                latent_model_input = latents.to(run_dtype)
                cfg_scale = high_sample_guide_scale
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
                    offload_device=offload_device,
                )
                # NOTE: DoNOT cast models to run_dtype, it will cause fp8 gemm error
                run_model = run_model.to(device)

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

        # if offload_model:
        #     gc.collect()
        #     torch.cuda.synchronize()
        # if dist.is_initialized():
        #     dist.barrier()
        log.debug(f"latents shape: {latents.shape}")
        # [bs, 16, fi, hi, wi]
        return latents

    @time_range
    def generate_video(
        self,
        prompt: str,
        prompt_negative: str = None,
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        device: torch.device = None,
        offload_device: torch.device = None,
    ):
        text_run_device = torch.device("cpu")  # TODO: maybe run text on cuda self.device
        positive, negative = self.forward_text_encoder(
            prompt, prompt_negative, device=text_run_device, offload_device=offload_device
        )
        latents = self.forward_diffusion_model(
            positive,
            negative,
            sample_config=sample_config,
            runtime_config=runtime_config,
            device=device,
            offload_device=offload_device,
        )
        if runtime_config.offload_model:
            self.offload_diffusion_model_to(offload_device)
        return latents

    @property
    def task_type(self):
        return self.pipeline_config.task_type

    @property
    def model_name(self):
        return self.pipeline_config.model_name

    @property
    def model_size(self):
        return self.pipeline_config.model_size

    @property
    def save_name(self):
        name = f"{self.pipeline_config.model_name}_{self.pipeline_config.task_type}_{self.pipeline_config.model_size}"
        name = name if not self.has_lora else name + "_with_lora"
        return name
