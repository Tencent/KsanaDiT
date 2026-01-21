import math
import random
import sys
from abc import abstractmethod

import torch
from tqdm import tqdm

from ..cache import create_hybrid_cache
from ..config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig, warp_as_hybrid_cache
from ..models import KsanaDiffusionModel, KsanaModelKey
from ..sample_solvers import get_sample_scheduler
from ..scheduler import KsanaBatchScheduler
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import evolve_with_recommend, log, time_range


class KsanaBaseGenerator(KsanaRunnerUnit):
    def __init__(self):
        super().__init__()
        self.batch_scheduler = KsanaBatchScheduler()

    @time_range
    def _preallocate_pinned_memory(self, diffusion_model, offload_device):
        # NOTE: preallocate pinned memory at warm up stage to avoid CPU OOM when merging lora
        for model in diffusion_model:
            if model:
                if hasattr(model, "preallocate_pinned_memory"):
                    model.preallocate_pinned_memory(offload_device)
                else:
                    raise ValueError(f"model {model} must have preallocate_pinned_memory method")

    def _valid_diffusion_model(
        self, diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel]
    ) -> list[KsanaDiffusionModel]:
        if isinstance(diffusion_model, (tuple, list)):
            diffusion_model = list(diffusion_model)
        elif isinstance(diffusion_model, KsanaDiffusionModel):
            diffusion_model = [diffusion_model]
        else:
            raise ValueError(
                f"diffusion_model {diffusion_model} must be KsanaDiffusionModel or list of KsanaDiffusionModel"
            )
        if len(diffusion_model) != 1:
            if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
                if len(diffusion_model) > 2 or len(diffusion_model) < 1:
                    raise ValueError(
                        f"{self.model_key} must have one or two model, but got {len(diffusion_model)} model"
                    )
                else:
                    if self.model_key != diffusion_model[0].model_key or self.model_key != diffusion_model[1].model_key:
                        raise ValueError(f"{self.model_key} must match but got {diffusion_model[0].model_key}")
                    if diffusion_model[0].run_dtype != diffusion_model[1].run_dtype:
                        raise ValueError(
                            f"{self.model_key} must have same run_dtype, but got "
                            f"{diffusion_model[0].run_dtype} and {diffusion_model[1].run_dtype}"
                        )
            else:
                raise ValueError(f"{self.model_key} must have only one model, but got {len(diffusion_model)} model")
        return diffusion_model

    def _valid_sample_config(self, sample_config: KsanaSampleConfig, model_len: int) -> KsanaSampleConfig:
        log.info(f"sample_config: {sample_config}")

        if isinstance(sample_config.cfg_scale, (float, int)):
            evolve_with_recommend(
                sample_config, {"cfg_scale": [sample_config.cfg_scale] * model_len}, force_update=True
            )
        elif isinstance(sample_config.cfg_scale, (list, tuple)):
            if len(sample_config.cfg_scale) != model_len:
                raise ValueError(f"cfg_scale length must be {model_len}, but got {len(sample_config.cfg_scale)}")
            evolve_with_recommend(sample_config, {"cfg_scale": list(sample_config.cfg_scale)}, force_update=True)
        else:
            raise TypeError(f"sample_config.cfg_scale {sample_config.cfg_scale} type not supported")

        if sample_config.denoise <= 0.0:
            raise ValueError(f"denoise <= 0.0 is not supported, got {sample_config.denoise}")
        return sample_config

    def _valid_cache_config(
        self, cache_config: KsanaCacheConfig | KsanaHybridCacheConfig, model_len: int
    ) -> KsanaHybridCacheConfig:
        log.info(f"cache_config: {cache_config}")

        if not (len(cache_config) == 1 or len(cache_config) == model_len):
            raise ValueError(f"cache_config length must be {model_len} or 1, but got {len(cache_config)}")
        hybrid_caches = []
        for i in range(model_len):
            cache_id = min(i, len(cache_config) - 1)  # allow two model use same cache config
            one_config = cache_config[cache_id]
            if one_config is None:
                hybrid_caches.append(None)
                continue
            if not isinstance(one_config, (KsanaCacheConfig, KsanaHybridCacheConfig)):
                raise ValueError(f"cache_config {one_config} must be KsanaCacheConfig or KsanaHybridCacheConfig")
            as_hybrid_cache = warp_as_hybrid_cache(one_config)
            hybrid_caches.append(as_hybrid_cache)
        return hybrid_caches

    def _valid_runtime_config(self, runtime_config: KsanaRuntimeConfig, num_prompts: int) -> KsanaRuntimeConfig:
        log.info(f"runtime_config: {runtime_config}")
        if runtime_config is None:
            raise ValueError("runtime_config must be provided")
        batch_size_per_prompts = runtime_config.batch_size_per_prompts
        if batch_size_per_prompts is None:
            batch_size_per_prompts = [1] * num_prompts
        elif isinstance(batch_size_per_prompts, int):
            batch_size_per_prompts = [batch_size_per_prompts] * num_prompts
        elif isinstance(batch_size_per_prompts, (list, tuple)):
            if len(batch_size_per_prompts) != num_prompts:
                raise ValueError(
                    f"batch_size_per_prompts({batch_size_per_prompts}) len " f"must match num_prompts ({num_prompts})"
                )
        else:
            raise TypeError(
                f"batch_size_per_prompts must be int/list[int]/None, but got {type(batch_size_per_prompts)}"
            )
        runtime_config = evolve_with_recommend(
            runtime_config,
            {"batch_size_per_prompts": batch_size_per_prompts},
            force_update=True,
        )
        return runtime_config

    def _valid_prompts(self, positive: torch.Tensor, negative: torch.Tensor) -> list[str]:
        log.info(
            f"positive shape:{positive.shape}, dtype:{positive.dtype}, device:{positive.device};"
            f" negtive shape:{negative.shape}, dtype:{negative.dtype}, device:{negative.device}"
        )
        if not (positive.ndim == negative.ndim == 3):
            raise ValueError(f"positive.shape {positive.shape}, negative.shape {negative.shape} must be 3D tensor")
        if positive.shape[0] != negative.shape[0]:
            raise ValueError(f"positive.shape[0] of {positive.shape}, negative.shape[0] of {negative.shape} must equal")
        return positive, negative

    def _expand_to_total_prompts_size(self, tensor: torch.Tensor, batch_size_per_prompts: list[int]):
        num_prompts = tensor.shape[0]
        total_prompts_num = sum(batch_size_per_prompts)
        if num_prompts > total_prompts_num:
            raise ValueError(f"total_prompts_num({total_prompts_num}) must >= num_prompts({num_prompts})")
        if total_prompts_num > num_prompts:
            repeats = torch.tensor(batch_size_per_prompts, dtype=torch.int64, device=tensor.device)
            tensor = tensor.repeat_interleave(repeats, dim=0)
        return tensor

    def _valid_image_to_total_prompts_size(
        self, img_latents: torch.Tensor, num_prompts: int, batch_size_per_prompts: list[int]
    ):
        if img_latents is None:
            return
        if num_prompts > img_latents.shape[0]:
            # Note: load img only get one tensor for each image to save memory, so batchsize at each prompt need repeats
            repeats_num = num_prompts // img_latents.shape[0]
            if repeats_num * img_latents.shape[0] != num_prompts:
                raise ValueError(
                    f"repeats_num({repeats_num}) * img_latents.shape{img_latents.shape}[0] must be equal"
                    f" num_prompts {num_prompts}"
                )
            img_latents = img_latents.repeat_interleave(repeats_num, dim=0)
        img_latents = self._expand_to_total_prompts_size(img_latents, batch_size_per_prompts)
        return img_latents

    def _valid_promots_to_total_prompts_size(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        batch_size_per_prompts: list[int],
    ):
        if positive.shape[0] != negative.shape[0]:
            raise ValueError(f"positive.shape[0] of {positive.shape} must equal negative.shape[0] of {negative.shape}")
        positive = self._expand_to_total_prompts_size(positive, batch_size_per_prompts)
        negative = self._expand_to_total_prompts_size(negative, batch_size_per_prompts)
        return positive, negative

    def _cast_to(self, src, *, dtype: torch.dtype, device: torch.device):
        if src.dtype != dtype:
            src = src.to(dtype)
        if src.device != device:
            src = src.to(device)
        return src

    def _create_random_noise_latents(
        self,
        total_samples_num: int,
        noise_shape: tuple[int],
        runtime_config: KsanaRuntimeConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        return tensor shape :[bs, z_dim, f, h, w] (5D tensor for batch)
        """
        seed = (
            runtime_config.seed
            if runtime_config.seed is not None and runtime_config.seed >= 0
            else random.randint(0, sys.maxsize)
        )
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        latents_list = []
        for _ in range(total_samples_num):
            single_noise = torch.randn(
                *noise_shape,
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            ).to(dtype)
            latents_list.append(single_noise)
        noise = torch.stack(latents_list, dim=0)
        log.info(f"create random noise_latents shape {noise.shape}, dtype:{noise.dtype}, device:{noise.device}")
        return noise, seed_g

    def _create_cache(self, cache_config: list[KsanaCacheConfig | None], model_key: KsanaModelKey):
        if cache_config is None:
            return None
        cache = []
        for config in cache_config:
            if config is None:
                cache.append(None)
                continue
            cache.append(create_hybrid_cache(model_key, config))
        return cache

    def _apply_rope_function_to_models(self, diffusion_models: list[KsanaDiffusionModel], rope_function: str | None):
        rope_value = rope_function or "default"
        for model in diffusion_models:
            if hasattr(model.model, "set_rope_function"):
                model.model.set_rope_function(rope_value)

    def _get_num_train_timesteps(self, default_settings):
        num_train_timesteps = getattr(default_settings.sample_config, "num_train_timesteps", None)
        if num_train_timesteps is None:
            raise RuntimeError("num_train_timesteps should be set in yaml sample_config settings")
        return num_train_timesteps

    def _use_cfg(self, cfg_scale: float, eps: float = 1e-6):
        return abs(cfg_scale - 1.0) > eps

    def run_one_batch(
        self,
        diffusion_model: list[KsanaDiffusionModel],
        positive: torch.Tensor | tuple,
        negative: torch.Tensor | tuple,
        noise_latent: torch.Tensor,
        img_latent: torch.Tensor,
        process_info: list[int],
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig],
        combine_cond_uncond: bool,
        timesteps: torch.Tensor,  # Tensor(list[int])
        run_dtype: torch.dtype,
        sample_scheduler_step_func,
        seed_g: torch.Generator,
        device: torch.device,
        offload_device: torch.device = None,
        comfy_bar_callback=None,
    ) -> torch.Tensor:
        log.info(f"timesteps:{timesteps}, combine_cond_uncond:{combine_cond_uncond}")
        dit_cache = self._create_cache(cache_config, self.model_key)

        cur_batch_size = self._get_num_prompts(positive)
        for iter_id, t in enumerate(tqdm(timesteps)):
            noise_latent = noise_latent.to(run_dtype)
            timestep = t.repeat(cur_batch_size)
            timestep_id = t.item()
            running_model = self.get_running_model(
                diffusion_model, timestep_id=timestep_id, device=device, offload_device=offload_device
            )
            running_cache = self.get_running_cache(dit_cache, timestep_id=timestep_id)
            cfg_scale = self.get_running_cfg_scale(cfg_scale=sample_config.cfg_scale, timestep_id=timestep_id)
            forward_kargs = self.prepare_model_forward_kargs(
                cfg_scale,
                diffusion_model=diffusion_model,
                noise_latent=noise_latent,
                timestep=timestep,
                combine_cond_uncond=combine_cond_uncond,
                step_iter=iter_id,
                cache=running_cache,
                positive=positive,
                negative=negative,
                img_latent=img_latent,
            )
            if self._use_cfg(cfg_scale):
                if combine_cond_uncond:
                    noise_pred_batch = running_model.forward(**forward_kargs)
                    noise_pred_cond, noise_pred_uncond = noise_pred_batch.chunk(2, dim=0)
                else:
                    if not isinstance(forward_kargs, (tuple, list)) or len(forward_kargs) != 2:
                        raise ValueError(f"forward_kargs {forward_kargs} must be tuple of (arg_cond, arg_uncond)")
                    arg_cond, arg_uncond = forward_kargs
                    noise_pred_cond = running_model.forward(**arg_cond)
                    noise_pred_uncond = running_model.forward(**arg_uncond)
                noise_pred = self.apply_cfg(cfg_scale, noise_pred_cond, noise_pred_uncond)
            else:
                noise_pred = running_model.forward(**arg_cond)

            step_out = sample_scheduler_step_func(noise_pred, t, noise_latent, return_dict=False, generator=seed_g)
            noise_latent = step_out if sample_config.solver == KsanaSolverType.EULER else step_out[0]
            if comfy_bar_callback is not None:
                steps = sample_config.steps
                batch_idx, num_batches = process_info
                current_step_iter = batch_idx * steps + (iter_id + 1)
                comfy_bar_callback(current_step_iter, num_batches * steps)

        [cache.show_cache_rate() if cache is not None else None for cache in dit_cache]
        return noise_latent

    def for_batches(
        self,
        batch_strategy,
        *,
        diffusion_model: list[KsanaDiffusionModel],
        noise_latents: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        img_latents: torch.Tensor,
        run_steps_kwargs: dict,
    ):
        sample_config = run_steps_kwargs["sample_config"]
        device = run_steps_kwargs["device"]

        log.info(f"batch_strategy={batch_strategy}")
        num_batches = len(batch_strategy)
        default_settings = diffusion_model[0].default_settings
        num_train_timesteps = self._get_num_train_timesteps(default_settings)
        for batch_idx, strategy_item in enumerate(batch_strategy):
            log.info(
                f"batch_idx {batch_idx}/{num_batches}, "
                f"strategy start={strategy_item.start}, end={strategy_item.end}, "
                f"combine_cond_uncond={strategy_item.combine_cond_uncond}"
            )
            batch_positive = self._split_tensors(positive, strategy_item.start, strategy_item.end)
            batch_negative = self._split_tensors(negative, strategy_item.start, strategy_item.end)
            batch_noise_latent = self._split_tensors(noise_latents, strategy_item.start, strategy_item.end)
            batch_img_latent = self._split_tensors(img_latents, strategy_item.start, strategy_item.end)

            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=num_train_timesteps, sample_config=sample_config, device=device
            )
            with torch.no_grad():
                processed_latents = self.run_one_batch(
                    process_info=[batch_idx, num_batches],
                    diffusion_model=diffusion_model,
                    sample_config=sample_config,
                    positive=batch_positive,
                    negative=batch_negative,
                    noise_latent=batch_noise_latent,
                    img_latent=batch_img_latent,
                    timesteps=timesteps,
                    sample_scheduler_step_func=sample_scheduler.step,
                    combine_cond_uncond=strategy_item.combine_cond_uncond,
                    **run_steps_kwargs,
                )

            noise_latents[strategy_item.start : strategy_item.end] = processed_latents
        return noise_latents

    @time_range
    def run(
        self,
        diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel],
        positive: torch.Tensor | tuple,  # [bs, text_len:512, 4096]
        negative: torch.Tensor | tuple,  # [bs, text_len:512, 4096]
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        noise_shape: list[
            int
        ],  # Note: noise shape do not include batch size :[vae_z_dim, lat_f, lat_h, lat_w] or [vae_z_dim, h, w]
        img_latents: torch.Tensor = None,  # [bs, vae_z_dim+4, lat_f, lat_h, lat_w]
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
        device=None,
        offload_device=None,
        comfy_bar_callback=None,
    ) -> torch.Tensor:
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_config (KsanaSampleConfig): _description_
        Returns:
            latents (torch.Tensor)
        """
        diffusion_model = self._valid_diffusion_model(diffusion_model, KsanaDiffusionModel)
        positive = self.preprocess_text_conditioning(positive)
        negative = self.preprocess_text_conditioning(negative)
        img_latents = self.preprocess_image_latent(img_latents)
        num_prompts = self._get_num_prompts(positive)

        sample_config = self._valid_sample_config(sample_config, len(diffusion_model))
        cache_config = self._valid_cache_config(cache_config, len(diffusion_model))
        runtime_config = self._valid_runtime_config(runtime_config, num_prompts)
        positive, negative = self._valid_prompts(positive, negative)

        noise_shape = self.valid_noise_shape(noise_shape, diffusion_model)

        self._apply_rope_function_to_models(diffusion_model, runtime_config.rope_function)
        self._preallocate_pinned_memory(diffusion_model, offload_device)

        # expand img_latents, positive and negative to total batch size supporting batch_size_per_prompts
        img_latents = self._valid_image_to_total_prompts_size(
            img_latents, num_prompts, runtime_config.batch_size_per_prompts
        )
        positive, negative = self._valid_promots_to_total_prompts_size(
            positive, negative, runtime_config.batch_size_per_prompts
        )
        run_dtype = diffusion_model[0].run_dtype
        positive, negative = self.cast_text_tensors_to(positive, negative, dtype=run_dtype, device=device)
        img_latents = self.cast_image_tensor_to(img_latents, dtype=run_dtype, device=device)

        # create noise latents and batch strategy
        total_samples_num = sum(runtime_config.batch_size_per_prompts)
        noise_latents, seed_g = self._create_random_noise_latents(
            total_samples_num, noise_shape, runtime_config, device=device, dtype=run_dtype
        )
        batch_strategy = self.batch_scheduler.build_batch_strategy(
            self.model_key, noise_latents.shape, total_samples_num, run_dtype, device
        )
        # Note: pack need after build strategy since strategy use noise_latents shape as 5D tensor
        noise_latents = self.pack_noise_latents(noise_latents)
        log.info(
            f"num_prompts: {num_prompts}, batch_size_per_prompts: {runtime_config.batch_size_per_prompts}, "
            f"total_samples_num: {total_samples_num} split as {len(batch_strategy)} batches"
        )
        run_steps_kwargs = {
            "sample_config": sample_config,
            "cache_config": cache_config,
            "runtime_config": runtime_config,
            "seed_g": seed_g,
            "run_dtype": run_dtype,
            "device": device,
            "offload_device": offload_device,
            "comfy_bar_callback": comfy_bar_callback,
        }
        noise_latents = self.for_batches(
            batch_strategy,
            diffusion_model=diffusion_model,
            positive=positive,
            negative=negative,
            img_latents=img_latents,
            run_steps_kwargs=run_steps_kwargs,
        )
        noise_latents = self.unpack_noise_latents(noise_latents)

        # TODO(qian): add auto estimate memory for automatic loading for all models
        if offload_device:
            [model.to(offload_device) for model in diffusion_model]

        # Note: [total_samples_num, vae_z_dim, f, h, w] or [total_samples_num, vae_z_dim, h, w]
        return noise_latents

    def _split_tensors(tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor], start: int, end: int):
        # tensor: torch.Tensor or tuple/list of torch.Tensor
        # return: sliced tensor with same type as input
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor[start:end]
        elif isinstance(tensor, (tuple, list)):
            return type(tensor)([t[start:end] for t in tensor])
        else:
            raise ValueError("tensor must be torch.Tensor or tuple/list of torch.Tensor")

    def _get_num_prompts(self, text_tensor: torch.Tensor | tuple):
        if isinstance(text_tensor, tuple):
            text_tensor = text_tensor[0]
        if isinstance(text_tensor, torch.Tensor):
            return text_tensor.shape[0]
        else:
            raise ValueError("text_tensor must be torch.Tensor or tuple of torch.Tensor")

    # ########## below are helper functions for subclass override ##########
    def preprocess_image_latent(self, img_latent):
        return img_latent

    def preprocess_text_conditioning(self, text_conditioning: torch.Tensor | tuple):
        return text_conditioning

    def valid_noise_shape(self, noise_shape: tuple[int] | list[int], **_):
        log.info(f"input noise_shape: {noise_shape}")
        if not isinstance(noise_shape, (tuple, list)):
            raise ValueError(f"noise_shape {noise_shape} must be tuple or list")
        noise_shape = list(noise_shape)

        if len(noise_shape) != 4:
            raise ValueError(
                f"{self.model_key} noise_shape {noise_shape} dim must "
                "be 4 like:[vae_z_dim:16, f, h, w], f==1 when generate image"
            )
        return noise_shape

    def pack_noise_latents(self, noise_latents: torch.Tensor) -> torch.Tensor:
        return noise_latents

    def unpack_noise_latents(self, noise_latents: torch.Tensor) -> torch.Tensor:
        return noise_latents

    def cast_text_tensors_to(self, positive, negative, *, dtype: torch.dtype, device: torch.device):
        positive = self._cast_to(positive, dtype=dtype, device=device)
        negative = self._cast_to(negative, dtype=dtype, device=device)
        return positive, negative

    def cast_image_tensor_to(self, img_latents, *, dtype: torch.dtype, device: torch.device):
        return self._cast_to(img_latents, dtype=dtype, device=device) if img_latents is not None else None

    def get_running_cfg_scale(self, cfg_scale: list[float], **kwargs):
        return cfg_scale[0] if isinstance(cfg_scale, (list, tuple)) else cfg_scale

    def get_running_model(self, diffusion_model: list[KsanaDiffusionModel], **kwargs):
        return diffusion_model[0] if isinstance(diffusion_model, (list, tuple)) else diffusion_model

    def get_running_cache(self, dit_cache: list, **kwargs):
        return dit_cache[0] if isinstance(dit_cache, (list, tuple)) else dit_cache

    def apply_cfg(self, cfg_scale, cond, uncond):
        return uncond + float(cfg_scale) * (cond - uncond)

    @abstractmethod
    def prepare_model_forward_kargs(self, cfg_scale, **kwargs) -> dict | tuple[dict, dict]:
        raise NotImplementedError("prepare_model_forward_kargs must be implemented in subclass")


@KsanaUnitFactory.register(KsanaUnitType.GENERATOR, [KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.Wan2_2_I2V_14B])
class KsanaWanGenerator(KsanaBaseGenerator):
    def __init__(self):
        super().__init__()
        # TODO: maybe could remove boundary, use allow each model input steps instead
        self.boundary = None

    def valid_noise_shape(self, noise_shape: tuple[int] | list[int], default_model_settings):
        noise_shape = super().valid_noise_shape(noise_shape, default_model_settings)
        if self.model_key == KsanaModelKey.Wan2_2_I2V_14B:
            # Note: i2v used img_latents as noise_shape, so need chanage to shape[1] as right z_dim
            #       and should have added z_dim to yaml settings
            if not hasattr(default_model_settings.vae, "z_dim"):
                raise ValueError("vae.z_dim not found in default_model_settings.vae")
            noise_shape[0] = default_model_settings.vae.z_dim
        return noise_shape

    @abstractmethod
    def cast_image_tensor_to(self, img_latents, *, dtype: torch.dtype, device: torch.device):
        if self.model_key == KsanaModelKey.Wan2_2_T2V_14B:
            return None
        return super().cast_image_tensor_to(img_latents, dtype=dtype, device=device)

    def get_model_boundary(self, diffusion_model: list[KsanaDiffusionModel]):
        if self.boundary is not None:
            return self.boundary

        if self.model_key not in [KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.Wan2_2_I2V_14B]:
            return None
        if len(diffusion_model) < 2:
            return None

        default_settings = diffusion_model[0].default_settings
        high_model, low_model = diffusion_model
        if low_model is not None:
            input_boundary = getattr(high_model.model_config, "boundary", None)
            default_boundary = getattr(default_settings.runtime_config, "boundary", None)
            boundary = input_boundary or default_boundary
            if boundary is None:
                raise RuntimeError("boundary should be set when low_model is not None")
            boundary = boundary * self._get_num_train_timesteps(default_settings)
            log.info(f"model boundary: {boundary}")
        else:
            boundary = None

    def get_running_model(self, diffusion_model, timestep_id: int, device=None, offload_device=None):
        assert device is not None, "device must be provided"
        self.boundary = self.get_model_boundary(diffusion_model)

        if len(diffusion_model) < 2:
            return None
        if not isinstance(diffusion_model, (float, int)):
            if diffusion_model.device != device:
                diffusion_model.to(device)
            return diffusion_model
        if len(diffusion_model) == 1:
            if diffusion_model[0].device != device:
                diffusion_model[0].to(device)
            return diffusion_model[0]
        if len(diffusion_model) != 2:
            raise ValueError(f"diffusion_model must be list of 1 or 2 float, but got {diffusion_model}")
        high_model, low_model = diffusion_model
        if low_model is not None and self.boundary is None:
            raise ValueError("boundary must be provided when low_model is not None")
        use_high = low_model is None or (self.boundary is not None and timestep_id >= self.boundary)
        if use_high:
            if low_model is not None:
                if low_model.device != offload_device:
                    low_model.to(offload_device)
            if high_model.device != device:
                high_model.to(device)
            return high_model
        else:
            if high_model.device != offload_device:
                high_model.to(offload_device)
            if low_model.device != device:
                low_model.to(device)
            return low_model

    def get_running_cache(self, dit_cache, timestep_id):
        if not isinstance(dit_cache, (float, int)):
            return dit_cache
        if len(dit_cache) == 1:
            return dit_cache[0]
        if len(dit_cache) != 2:
            raise ValueError(f"dit_cache must be list of 1 or 2 float, but got {dit_cache}")

        high_cache, low_cache = dit_cache
        if low_cache is None:
            return high_cache
        if timestep_id >= self.boundary:
            return high_cache
        else:
            high_cache.offload_to_cpu()
            return low_cache

    def get_running_cfg_scale(self, cfg_scale: list[float], timestep_id: int):
        if not isinstance(cfg_scale, (float, int)):
            return cfg_scale
        if len(cfg_scale) == 1:
            return cfg_scale[0]
        if len(cfg_scale) != 2:
            raise ValueError(f"cfg_scales must be list of 1 or 2 float, but got {cfg_scale}")
        if cfg_scale[1] is not None and self.boundary is not None and timestep_id < self.boundary:
            return cfg_scale[1]
        else:
            return cfg_scale[0]

    def _get_seq_len(self, target_shape, diffusion_model):
        default_settings = diffusion_model.default_settings
        if hasattr(default_settings.diffusion, "patch_size"):
            patch_size = default_settings.diffusion.patch_size
        else:
            raise ValueError(f"default_settings.diffusion {default_settings.diffusion} must have patch_size")
        sp_size = diffusion_model.sp_size
        _, _, lat_f, lat_h, lat_w = target_shape
        max_seq_len = (lat_f * lat_h * lat_w) // (patch_size[1] * patch_size[2])
        return int(math.ceil(max_seq_len / sp_size)) * sp_size

    def prepare_model_forward_kargs(
        self,
        cfg_scale,
        *,
        diffusion_model,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        positive,
        negative,
        img_latent,
    ) -> dict:
        seq_len = self._get_seq_len(noise_latent.shape, diffusion_model[0])
        base = {"cache": cache, "step_iter": step_iter}
        if self._use_cfg(cfg_scale) and combine_cond_uncond:
            # latent: [bs, z_dim, fi, hi, wi] => [2*bs, z_dim, fi, hi, wi]
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_context = torch.cat([positive, negative], dim=0)
            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "context": combine_context,
                "seq_len": seq_len,
            }
            if self.model_key == KsanaModelKey.Wan2_2_I2V_14B and img_latent is not None:
                combine_kargs["y"] = torch.cat([img_latent, img_latent], dim=0)
            return base | combine_kargs

        base.update({"x": noise_latent, "t": timestep})
        arg_cond = {"phase": "cond", "context": positive, "seq_len": seq_len}
        arg_uncond = {"phase": "uncond", "context": negative, "seq_len": seq_len}
        if self.model_key == KsanaModelKey.Wan2_2_I2V_14B:
            arg_cond["y"] = img_latent
            arg_uncond["y"] = img_latent
        return base | arg_cond, base | arg_uncond


@KsanaUnitFactory.register(KsanaUnitType.GENERATOR, KsanaModelKey.QwenImage_T2I)
class KsanaQwenGenerator(KsanaBaseGenerator):

    def __init__(self):
        super().__init__()
        self.vae_patch_size = None
        self.latent_img_shapes = None

    def preprocess_text_conditioning(self, conditioning: torch.Tensor | tuple) -> tuple:
        if isinstance(conditioning, (tuple, list)) and len(conditioning) == 2:
            embeds, mask = conditioning
            if isinstance(mask, torch.Tensor):
                return embeds, mask
        # ComfyUI format: only embeddings tensor provided
        if isinstance(conditioning, torch.Tensor):
            embeds = conditioning
            # Generate all-ones attention mask based on sequence length
            batch_size, seq_len = embeds.shape[:2]
            mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=embeds.device)
            return embeds, mask
        raise ValueError(f"Unsupported conditioning format: {type(conditioning)}")

    def _valid_prompts(self, positive: tuple, negative: tuple):
        pos, pos_mask = positive
        neg, neg_mask = negative
        log.info("text encoder tensor:")
        pos, neg = super()._valid_prompts(pos, neg)
        log.info("text mask:")
        pos_mask, neg_mask = super()._valid_prompts(pos_mask, neg_mask)
        return (pos, pos_mask), (neg, neg_mask)

    def _valid_promots_to_total_prompts_size(
        self,
        positive: tuple,
        negative: tuple,
        batch_size_per_prompts: list[int],
    ):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos, neg = super()._valid_promots_to_total_prompts_size(pos, neg, batch_size_per_prompts)
        pos_mask, neg_mask = super()._valid_promots_to_total_prompts_size(pos_mask, neg_mask, batch_size_per_prompts)
        neg = self._expand_to_total_prompts_size(neg, batch_size_per_prompts)
        pos_mask = self._expand_to_total_prompts_size(pos_mask, batch_size_per_prompts)
        neg_mask = self._expand_to_total_prompts_size(neg_mask, batch_size_per_prompts)
        return (pos, pos_mask), (neg, neg_mask)

    def cast_text_tensors_to(self, positive: tuple, negative: tuple, *, dtype: torch.dtype, device: torch.device):
        pos, pos_mask = positive
        neg, neg_mask = negative
        pos, neg = super().cast_text_tensors_to(pos, neg, dtype=dtype, device=device)
        return (pos, pos_mask), (neg, neg_mask)

    def calculate_shift(self, seq_len: int, configs) -> float:
        base_seq_len = getattr(configs, "base_seq_len", 256)
        max_seq_len = getattr(configs, "max_seq_len", 4096)
        base_shift = getattr(configs, "base_shift", 0.5)
        max_shift = getattr(configs, "max_shift", 1.15)

        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return seq_len * m + b

    def prepare_model_forward_kargs(
        self,
        cfg_scale,
        *,
        positive,
        negative,
        diffusion_model,
        noise_latent,
        timestep,
        combine_cond_uncond,
        step_iter,
        cache,
        img_latent,
    ) -> dict | tuple[dict, dict]:
        if cache is not None:
            raise NotImplementedError(f"{self.model_key} does not support cache yet!")
        base = {"cache": cache, "step_iter": step_iter}
        positive_embeds, positive_mask = positive
        negative_embeds, negative_mask = negative

        img_shapes = self._get_latent_img_shapes()

        positive_txt_seq_lens = positive_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_mask.sum(dim=1).tolist()
        # TODO(qian): in multi cards seqlen = seqlen // diffusion_model.sp_size
        if self._use_cfg(cfg_scale) and combine_cond_uncond:
            combine_x = torch.cat([noise_latent, noise_latent], dim=0)
            combine_t = torch.cat([timestep, timestep], dim=0)
            combine_embs = torch.cat([positive_embeds, negative_embeds], dim=0)
            combine_mask = torch.cat([positive_mask, negative_mask], dim=0)
            combine_txt_seq_lens = positive_txt_seq_lens + negative_txt_seq_lens
            combine_img_shapes = img_shapes + img_shapes
            combine_kargs = {
                "phase": "combine",
                "x": combine_x,
                "t": combine_t,
                "img_shapes": combine_img_shapes,
                "context": combine_embs,
                "context_mask": combine_mask,
                "txt_seq_lens": combine_txt_seq_lens,
            }
            return base | combine_kargs

        base.update({"x": noise_latent, "t": timestep, "img_shapes": img_shapes})
        arg_cond = {
            "phase": "cond",
            "context": positive_embeds,
            "context_mask": positive_mask,
            "txt_seq_lens": positive_txt_seq_lens,
        }
        arg_uncond = {
            "phase": "uncond",
            "context": negative,
            "context_mask": negative_mask,
            "txt_seq_lens": negative_txt_seq_lens,
        }
        return base | arg_cond, base | arg_uncond

    def _get_patch_size(self):
        if self.vae_patch_size is None:
            raise ValueError(f"{self.model_key} vae_patch_size is None, cannot get patch_size")
        return self.vae_patch_size

    def _get_latent_img_shapes(self):
        if self.latent_img_shapes is None:
            raise ValueError("latent_img_shapes is None, please call pack_noise_latents first to set it")
        return self.latent_img_shapes

    def pack_noise_latents(self, latents):
        patch_size = self._get_patch_size()
        if latents.dim() != 5:
            raise ValueError(f"{self.model_key} pack latents {latents.shape} must be 5D tensor")
        num, z_dim, latent_f, latent_h, latent_w = latents.shape
        if latent_f != 1:
            raise ValueError(f"{self.model_key} pack latents latent_f  must be 1, but got {latent_f}")
        latents = latents.squeeze(2)  # remove latent_f dim
        latents = latents.view(num, z_dim, latent_h // patch_size, patch_size, latent_w // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        new_h = latent_h // patch_size
        new_w = latent_w // patch_size
        latents = latents.reshape(num, new_h * new_w, z_dim * patch_size * patch_size)

        self.latent_img_shapes = [[(1, new_h, new_w)] for _ in range(num)]
        log.info(f"{self.model_key} pack noise latents to shape {latents.shape}")
        return latents

    def unpack_noise_latents(self, latents):
        if latents.dim() != 3:
            raise ValueError(f"{self.model_key} unpack latents input {latents.shape} must be 3D tensor")
        num, hw, z_dim = latents.shape
        patch_size = self._get_patch_size()
        img_shapes = self._get_latent_img_shapes()
        _, new_h, new_w = img_shapes[0]
        latents = latents.view(num, new_h, new_w, z_dim // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(num, z_dim // (patch_size * patch_size), 1, new_h * patch_size, new_w * patch_size)
        log.info(f"{self.model_key} unpack noise latents shape from {(num, hw, z_dim)} to {latents.shape}")
        return latents

    @time_range
    def run(
        self,
        diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel],
        sample_config: KsanaSampleConfig = None,
        **kwargs,
    ):
        model = diffusion_model[0] if isinstance(diffusion_model, (list, tuple)) else diffusion_model
        default_settings = model.default_settings
        self.vae_patch_size = getattr(default_settings.vae, "patch_size", 2)

        # TODO(TJ): calculate shift value based on seq_len
        # mu = self.calculate_shift(seq_len==newh*neww==packed_noise.shape[1], default_settings.sample_config)
        # sample_config = evolve_with_recommend(sample_config, {"shift": mu})

        latents = super().run(diffusion_model=diffusion_model, sample_config=sample_config, **kwargs)

        return latents
