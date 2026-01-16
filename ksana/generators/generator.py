import math
import random
import sys

import torch
from tqdm import tqdm

from ..cache import create_hybrid_cache
from ..config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from ..config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig, warp_as_hybrid_cache
from ..models import KsanaDiffusionModel, KsanaModelKey
from ..sample_solvers import calculate_shift, get_sample_scheduler
from ..scheduler import KsanaScheduler
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, print_recursive, time_range
from ..utils.profile import MemoryProfiler


class KsanaBaseGenerator(KsanaRunnerUnit):
    def __init__(self):
        super().__init__()
        self.scheduler = KsanaScheduler()

    @time_range
    def preallocate_pinned_memory(self, high_model, low_model, offload_device):
        # NOTE: preallocate pinned memory at warm up stage to avoid CPU OOM when merging lora
        for model in [high_model, low_model]:
            if model:
                model.preallocate_pinned_memory(offload_device)

    def change_to_hybrid_cache(
        self, cache_configs: list[KsanaCacheConfig | KsanaHybridCacheConfig], target_len: int
    ) -> list[KsanaHybridCacheConfig]:
        if cache_configs is None:
            return None
        if not (len(cache_configs) == 1 or len(cache_configs) == target_len):
            raise ValueError(f"cache_configs length must be {target_len} or 1, but got {len(cache_configs)}")
        hybrid_caches = []
        for i in range(target_len):
            cache_id = min(i, len(cache_configs) - 1)  # allow two model use same cache config
            cache_config = cache_configs[cache_id]
            as_hybrid_cache = warp_as_hybrid_cache(cache_config)
            hybrid_caches.append(as_hybrid_cache)
        return hybrid_caches

    def valid_args(self, diffusion_models: list[KsanaDiffusionModel], sample_config):
        high_model = diffusion_models
        low_model = None
        if hasattr(diffusion_models, "__len__"):
            if len(diffusion_models) == 1:
                high_model = diffusion_models[0]
            else:
                assert len(diffusion_models) <= 2, f"size of model must be 2, but got {len(diffusion_models)}"
                high_model, low_model = diffusion_models
        if isinstance(sample_config.cfg_scale, float):
            high_sample_guide_scale = sample_config.cfg_scale
            low_sample_guide_scale = None if low_model is None else sample_config.cfg_scale
        elif hasattr(sample_config.cfg_scale, "__len__"):
            assert (
                len(sample_config.cfg_scale) == 2
            ), f"size of cfg_scale must be 2, but got {len(sample_config.cfg_scale)}"
            high_sample_guide_scale = sample_config.cfg_scale[0]
            low_sample_guide_scale = sample_config.cfg_scale[1]
        else:
            raise ValueError(f"sample_config.cfg_scale {sample_config.cfg_scale} not supported")
        assert sample_config.denoise > 0.0, f"denoise <= 0.0 is not supported, got {sample_config.denoise}"
        return (
            high_model,
            low_model,
            high_sample_guide_scale,
            low_sample_guide_scale,
        )

    def cast_to(self, src, dtype: torch.dtype, device: torch.device):
        if src.dtype != dtype:
            src = src.to(dtype)
        if src.device != device:
            src = src.to(device)
        return src

    def _expand_conditioning_by_batch_size_per_prompts(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        batch_size_per_prompts: list[int],
        img_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        repeats = torch.tensor(batch_size_per_prompts, dtype=torch.int64, device=positive.device)
        positive = positive.repeat_interleave(repeats, dim=0)
        negative = negative.repeat_interleave(repeats, dim=0)

        if img_latents is not None:
            img_latents = img_latents.repeat_interleave(repeats, dim=0)

        return positive, negative, img_latents

    def _valid_noise_shape(self, noise_shape: tuple[int], default_model_settings):
        if len(noise_shape) != 4:
            raise ValueError(f"noise_shape {noise_shape} dim must be 4:[vae_z_dim:16, f, h, w]")
        if self.model_key.is_i2v_type():
            # Note: i2v used img_latents as noise_shape, so need chanage to shape[1] as right z_dim
            #       and should have added z_dim to yaml settings
            if not hasattr(default_model_settings.vae, "z_dim"):
                raise ValueError("vae.z_dim not found in default_model_settings.vae")
            noise_shape[0] = default_model_settings.vae.z_dim
        return noise_shape

    def _create_random_noise_latents(
        self,
        total_batch_size: int,
        noise_shape: tuple[int],
        runtime_config: KsanaRuntimeConfig,
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
        for _ in range(total_batch_size):
            single_noise = torch.randn(
                *noise_shape,
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            ).to(dtype)
            latents_list.append(single_noise)
        noise = torch.stack(latents_list, dim=0)
        return noise, seed_g

    def get_seq_len(self, target_shape, patch_size: list[int], sp_size: int):
        _, _, lat_f, lat_h, lat_w = target_shape
        max_seq_len = (lat_f * lat_h * lat_w) // (patch_size[1] * patch_size[2])
        return int(math.ceil(max_seq_len / sp_size)) * sp_size

    def _apply_rope_function_to_models(self, diffusion_models: list[KsanaDiffusionModel], rope_function: str | None):
        rope_value = rope_function or "default"
        for model in diffusion_models:
            if hasattr(model.model, "set_rope_function"):
                model.model.set_rope_function(rope_value)

    def get_run_model(self, high_model, low_model, timestep_id: int, boundary: float, device=None, offload_device=None):
        assert device is not None, "device must be provided"
        if low_model is not None and boundary is None:
            raise ValueError("boundary must be provided when low_model is not None")
        use_high = low_model is None or (boundary is not None and timestep_id >= boundary)
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

    def get_run_cache(self, high_cache, low_cache, timestep_id, boundary):
        if low_cache is None:
            return high_cache
        if timestep_id >= boundary:
            return high_cache
        else:
            high_cache.offload_to_cpu()
            return low_cache

    def use_cfg(self, cfg_scale: float, eps: float = 1e-6):
        return abs(cfg_scale - 1.0) > eps

    def run_steps_by_batch(
        self,
        positive_batch: torch.Tensor,
        negative_batch: torch.Tensor,
        noise_latents_batch: torch.Tensor,
        img_latents_batch: torch.Tensor,
        combine_cond_uncond: bool,
        seq_len: int,
        timesteps: torch.Tensor,
        run_dtype: torch.dtype,
        guide_scales: tuple[float, float],  # (high_sample_guide_scale, low_sample_guide_scale)
        models: tuple,  # (high_model, low_model)
        sample_scheduler_step_func,
        sample_config_solver_name: str,
        seed_g: torch.Generator,
        device: torch.device,
        boundary: float = None,
        offload_device: torch.device = None,
        cache_configs: tuple = None,  # (high_cache_config, low_cache_config)
        bar_info_callback: tuple = None,  # (bar_info, comfy_bar_callback)
    ) -> torch.Tensor:
        high_sample_guide_scale, low_sample_guide_scale = guide_scales
        high_model, low_model = models
        high_cache_config, low_cache_config = cache_configs if cache_configs else (None, None)
        bar_info, comfy_bar_callback = bar_info_callback if bar_info_callback else (None, None)

        # 每次重新创建cache
        low_cache = None
        high_cache = None
        if high_cache_config is not None:
            high_cache = create_hybrid_cache(
                high_model.model_key,
                high_cache_config,
            )
        if low_cache_config is not None and low_model is not None:
            low_cache = create_hybrid_cache(
                low_model.model_key,
                low_cache_config,
            )

        batch_size_current = positive_batch.shape[0]

        arg_cond = {"phase": "cond", "context": positive_batch, "seq_len": seq_len}
        arg_uncond = {"phase": "uncond", "context": negative_batch, "seq_len": seq_len}

        if self.model_key.is_i2v_type():
            arg_cond["y"] = img_latents_batch
            arg_uncond["y"] = img_latents_batch
        arg_combine = None
        if combine_cond_uncond:
            arg_combine = {
                "phase": "combine",
                "context": torch.cat([positive_batch, negative_batch], dim=0),
                "seq_len": seq_len,
            }
            if self.model_key.is_i2v_type() and img_latents_batch is not None:
                arg_combine["y"] = torch.cat([img_latents_batch, img_latents_batch], dim=0)

        total_steps = len(timesteps)
        for iter_id, t in enumerate(tqdm(timesteps)):
            MemoryProfiler.record_memory(f"before_inference_loop_iter_{iter_id}")
            latent_model_input = noise_latents_batch.to(run_dtype)
            cfg_scale = high_sample_guide_scale
            if low_model is not None and boundary is not None and t.item() < boundary:
                cfg_scale = low_sample_guide_scale

            timestep = t.repeat(batch_size_current)
            timestep_id = t.item()

            run_model = self.get_run_model(
                high_model=high_model,
                low_model=low_model,
                timestep_id=timestep_id,
                boundary=boundary,
                device=device,
                offload_device=offload_device,
            )

            MemoryProfiler.record_memory(f"inference_step_{iter_id}_after_model_switch")
            run_cache = self.get_run_cache(
                high_cache=high_cache,
                low_cache=low_cache,
                timestep_id=timestep_id,
                boundary=boundary,
            )

            if self.use_cfg(cfg_scale):
                if combine_cond_uncond:
                    # latent: [bs, 16, fi, hi, wi] => [2*bs, 16, fi, hi, wi]
                    latent_batch = torch.cat([latent_model_input, latent_model_input], dim=0)
                    # timestep: [bs] => [2*bs]
                    timestep_batch = torch.cat([timestep, timestep], dim=0)
                    noise_pred_batch = run_model.forward(
                        x=latent_batch, t=timestep_batch, step_iter=iter_id, cache=run_cache, **arg_combine
                    )
                    # 分离结果: [2*bs, 16, fi, hi, wi] => 2 x [bs, 16, fi, hi, wi]
                    noise_pred_cond, noise_pred_uncond = noise_pred_batch.chunk(2, dim=0)
                else:
                    noise_pred_uncond = run_model.forward(
                        x=latent_model_input, t=timestep, step_iter=iter_id, cache=run_cache, **arg_uncond
                    )
                    noise_pred_cond = run_model.forward(
                        x=latent_model_input, t=timestep, step_iter=iter_id, cache=run_cache, **arg_cond
                    )
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = run_model.forward(
                    x=latent_model_input, t=timestep, step_iter=iter_id, cache=run_cache, **arg_cond
                )

            temp_x0 = sample_scheduler_step_func(
                noise_pred,
                t,
                noise_latents_batch,
                return_dict=False,
                generator=seed_g,
            )
            noise_latents_batch = temp_x0 if sample_config_solver_name == KsanaSolverType.EULER else temp_x0[0]
            MemoryProfiler.record_memory(f"inference_step_{iter_id}_after_sample_scheduler")
            if comfy_bar_callback is not None:
                if bar_info is not None:
                    batch_step_offset, global_total_steps = bar_info
                    # 当前全局步骤 = batch起始步骤 + 当前batch内步骤
                    global_current_step = batch_step_offset + (iter_id + 1)
                    comfy_bar_callback(global_current_step, global_total_steps)
                else:
                    # 兼容原有逻辑
                    comfy_bar_callback(iter_id + 1, total_steps)

        if high_cache is not None:
            high_cache.show_cache_rate()
        if low_cache is not None:
            low_cache.show_cache_rate()

        return noise_latents_batch


@KsanaUnitFactory.register(KsanaUnitType.GENERATOR, [KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.Wan2_2_I2V_14B])
class KsanaWanGenerator(KsanaBaseGenerator):

    @time_range
    def run(
        self,
        diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel],
        positive: torch.Tensor,  # [bs, text_len:512, 4096]
        negative: torch.Tensor,  # [bs, text_len:512, 4096]
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        noise_shape: list[int],  # Note: noise shape do not include batch size :[vae_z_dim, lat_f, lat_h, lat_w]
        img_latents: torch.Tensor = None,  # [bs, vae_z_dim+4, lat_f, lat_h, lat_w]
        cache_configs: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
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
        log.info(f"runtime_config: {runtime_config}")
        log.info(f"sample_config: {sample_config}")
        log.info(f"cache_configs: {cache_configs}")
        log.info(f"input noise_shape: {noise_shape}")
        high_cache_config, low_cache_config = None, None
        if not isinstance(diffusion_model, list):
            diffusion_model = [diffusion_model]
        cache_configs = self.change_to_hybrid_cache(cache_configs, len(diffusion_model))
        if cache_configs is not None:
            high_cache_config = cache_configs[0]
            low_cache_config = cache_configs[1] if len(cache_configs) > 1 else None

        high_model, low_model, high_sample_guide_scale, low_sample_guide_scale = self.valid_args(
            diffusion_model, sample_config
        )
        log.info(
            f"high_sample_guide_scale: {high_sample_guide_scale}, low_sample_guide_scale: {low_sample_guide_scale}"
        )
        assert (
            low_model is None or high_model.run_dtype == low_model.run_dtype
        ), f"high_model.run_dtype {high_model.run_dtype}, low_model.run_dtype {low_model.run_dtype} should be same"
        run_dtype = high_model.run_dtype
        self._apply_rope_function_to_models(diffusion_model, runtime_config.rope_function)
        self.preallocate_pinned_memory(high_model, low_model, offload_device)

        log.debug("positive, negtive:")
        print_recursive(positive, log.debug)
        print_recursive(negative, log.debug)
        # [bs, input_text_len, 4096]
        assert positive.ndim == negative.ndim == 3, f"positive.shape {positive.shape}, negative.shape {negative.shape}"
        positive = self.cast_to(positive, run_dtype, device)
        negative = self.cast_to(negative, run_dtype, device)

        img_latents = (
            self.cast_to(img_latents, run_dtype, device)
            if img_latents is not None and self.model_key.is_i2v_type()
            else None
        )

        if img_latents is not None and positive.shape[0] > img_latents.shape[0]:
            # Note: ont img only get one tensor for saving memory, so batchsize at each prompt need repeats
            repeats_num = positive.shape[0] // img_latents.shape[0]
            img_latents = img_latents.repeat_interleave(repeats_num, dim=0)

        num_prompts = positive.shape[0]
        batch_size_per_prompts = runtime_config.batch_size_per_prompts
        total_batch_size = sum(batch_size_per_prompts)
        if total_batch_size != num_prompts:
            # TODO: this is not work for batch_size_per_prompts: like [1,3,2]
            positive, negative, img_latents = self._expand_conditioning_by_batch_size_per_prompts(
                positive, negative, img_latents=img_latents, batch_size_per_prompts=batch_size_per_prompts
            )

        default_model_settings = high_model.default_settings
        noise_shape = self._valid_noise_shape(noise_shape, default_model_settings)
        noise_latents, seed_g = self._create_random_noise_latents(
            total_batch_size, noise_shape, runtime_config, device, run_dtype
        )
        log.info(f"total noise_shape: {noise_latents.shape}")

        seq_len = self.get_seq_len(noise_latents.shape, default_model_settings.diffusion.patch_size, high_model.sp_size)

        # TODO(TJ): optmize boundary, not all model need boundary
        boundary = None
        if self.model_key in [KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.Wan2_2_I2V_14B]:
            if low_model is not None:
                boundary = (
                    high_model.model_config.boundary
                    if high_model.model_config.boundary
                    else default_model_settings.runtime_config.boundary
                )
                if boundary is None:
                    raise RuntimeError("boundary should be set when low_model is not None")
                boundary = boundary * default_model_settings.sample_config.num_train_timesteps
            else:
                boundary = None

        with torch.no_grad():
            batch_strategy = self.scheduler.build_batch_strategy(
                high_model.model_key, noise_latents.shape, total_batch_size, run_dtype, device
            )

            # 计算全局进度信息
            total_steps_per_batch = sample_config.steps
            global_total_steps = len(batch_strategy) * total_steps_per_batch
            log.info(f"batch_strategy={batch_strategy}")
            # 使用动态batch处理
            for batch_idx, strategy_item in enumerate(batch_strategy):
                pos_batch = positive[strategy_item.start : strategy_item.end]
                neg_batch = negative[strategy_item.start : strategy_item.end]
                noise_latents_batch = noise_latents[strategy_item.start : strategy_item.end]
                img_latents_batch = (
                    img_latents[strategy_item.start : strategy_item.end] if img_latents is not None else None
                )
                log.info(
                    f"batch {batch_idx} strategy start = {strategy_item.start}, end = {strategy_item.end} "
                    f"combine = {strategy_item.combine_cond_uncond}"
                )
                MemoryProfiler.record_memory(f"batch_{strategy_item.start}-{strategy_item.end}_before_inference_loop")

                # 计算当前batch的起始步骤偏移
                batch_step_offset = batch_idx * total_steps_per_batch
                bar_info = (batch_step_offset, global_total_steps)

                batch_sample_scheduler, _, batch_timesteps = get_sample_scheduler(
                    num_train_timesteps=default_model_settings.sample_config.num_train_timesteps,
                    sampling_steps=sample_config.steps,
                    sample_solver=sample_config.solver,
                    device=device,
                    shift=sample_config.shift,
                    denoise=sample_config.denoise,
                    sigmas=sample_config.sigmas,
                )
                log.info(f"batch timesteps: {batch_timesteps}, boundary:{boundary}, seq_len:{seq_len}")

                # TODO(TJ): optimize me
                processed_latents = self.run_steps_by_batch(
                    positive_batch=pos_batch,
                    negative_batch=neg_batch,
                    noise_latents_batch=noise_latents_batch,
                    img_latents_batch=img_latents_batch,
                    combine_cond_uncond=strategy_item.combine_cond_uncond,
                    seq_len=seq_len,
                    timesteps=batch_timesteps,
                    run_dtype=run_dtype,
                    guide_scales=(high_sample_guide_scale, low_sample_guide_scale),
                    models=(high_model, low_model),
                    sample_scheduler_step_func=batch_sample_scheduler.step,
                    sample_config_solver_name=sample_config.solver,
                    seed_g=seed_g,
                    device=device,
                    boundary=boundary,
                    offload_device=offload_device,
                    cache_configs=(high_cache_config, low_cache_config),
                    bar_info_callback=(bar_info, comfy_bar_callback),
                )
                MemoryProfiler.record_memory(f"batch_{strategy_item.start}-{strategy_item.end}_after_inference_loop")

                noise_latents[strategy_item.start : strategy_item.end] = processed_latents

        log.debug(f"noise_latents shape: {noise_latents.shape}")

        # TODO: estimate diffusion memory usage to check whether neeed to offload diffusion model
        # if runtime_config.offload_model and offload_device is not None :
        # here always offload diffusion model to offload_device
        if offload_device:
            [model.to(offload_device) for model in diffusion_model]

        # [bs, vae_z_dim, f, h, w]
        return noise_latents


@KsanaUnitFactory.register(KsanaUnitType.GENERATOR, KsanaModelKey.QwenImage_T2I)
class KsanaQwenGenerator(KsanaBaseGenerator):
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape  # pylint: disable=unused-variable
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, height, width)
        return latents

    def create_image_latents(
        self,
        width: int,
        height: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
    ):
        latent_h = 2 * (height // (self.vae_scale_factor * 2))
        latent_w = 2 * (width // (self.vae_scale_factor * 2))
        num_channels = self.vae_z_dim

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        latents = torch.randn(
            batch_size,
            1,
            num_channels,
            latent_h,
            latent_w,  # 5D: (B, T=1, C, H, W)
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(dtype)

        latents = self._pack_latents(latents, batch_size, num_channels, latent_h, latent_w)
        return latents, latent_h, latent_w

    def _parse_conditioning_input(self, conditioning):
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

    def _infer_image_size_from_latents(self, img_latents: torch.Tensor, runtime_config: KsanaRuntimeConfig):
        if runtime_config is not None and runtime_config.size is not None:
            size = runtime_config.size
            if isinstance(size, (tuple, list)) and len(size) == 2:
                return size[0], size[1]

        if img_latents is not None:
            if img_latents.dim() == 4:
                # [B, C, H, W] - SD3/Qwen-Image ComfyUI format
                _, _, lat_h, lat_w = img_latents.shape
            elif img_latents.dim() == 5:
                # [B, C, T, H, W] - video format
                _, _, _, lat_h, lat_w = img_latents.shape
            else:
                raise ValueError(f"Unsupported latent tensor shape: {img_latents.shape}")
            height = lat_h * self.vae_scale_factor
            width = lat_w * self.vae_scale_factor
            return width, height

        default_config = self.pipeline_config.default_config
        if hasattr(default_config, "size") and default_config.size is not None:
            return default_config.size

        raise ValueError("Cannot infer image size: no latents or runtime_config.size provided")

    # TODO(TJ): use same or abstrct some template
    @time_range
    def run(
        self,
        diffusion_models,
        positive,  # (embeds, mask) or tensor (ComfyUI format)
        negative,  # (embeds, mask) or tensor (ComfyUI format)
        img_latents=None,  # [B, 16, H, W] from ComfyUI EmptySD3LatentImage
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        device=None,
        offload_device=None,
        comfy_bar_callback=None,
        **kwargs,
    ):
        log.info(f"sample_config: {sample_config}, runtime_config: {runtime_config}")

        model = diffusion_models[0] if isinstance(diffusion_models, (list, tuple)) else diffusion_models
        run_dtype = model.run_dtype
        if offload_device is not None:
            self.preallocate_pinned_memory(model, None, offload_device)
        positive_embeds, positive_mask = self._parse_conditioning_input(positive)
        negative_embeds, negative_mask = self._parse_conditioning_input(negative)

        width, height = self._infer_image_size_from_latents(img_latents, runtime_config)
        log.info(f"Inferred image size: width={width}, height={height}")

        seed = (
            runtime_config.seed
            if runtime_config is not None and runtime_config.seed is not None
            else random.randint(0, sys.maxsize)
        )

        batch_size = positive_embeds.shape[0]
        latents, latent_h, latent_w = self.create_image_latents(width, height, seed, device, run_dtype, batch_size)

        img_shapes = [[(1, latent_h // 2, latent_w // 2)] for _ in range(batch_size)]
        positive_txt_seq_lens = positive_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = negative_mask.sum(dim=1).tolist()

        seq_len = latents.shape[1]
        default_config = self.pipeline_config.default_config

        # Use shift from sample_config if provided, otherwise calculate dynamically
        if sample_config.shift is not None:
            mu = sample_config.shift
        else:
            mu = calculate_shift(
                seq_len,
                default_config.base_seq_len,
                default_config.max_seq_len,
                default_config.base_shift,
                default_config.max_shift,
            )

        sample_scheduler, _, timesteps = get_sample_scheduler(
            num_train_timesteps=default_config.sample_config.num_train_timesteps,
            sampling_steps=sample_config.steps,
            sample_solver=sample_config.solver,
            device=device,
            shift=mu,
            denoise=sample_config.denoise,
        )

        latents = latents.to(dtype=run_dtype, device=device)
        positive_embeds = positive_embeds.to(dtype=run_dtype, device=device)
        negative_embeds = negative_embeds.to(dtype=run_dtype, device=device)

        cfg_scale = sample_config.cfg_scale
        if isinstance(cfg_scale, (tuple, list)):
            cfg_scale = float(cfg_scale[0])
        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6

        total_steps = len(timesteps)
        model.to(device)
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                timestep = t.unsqueeze(0).to(run_dtype)
                noise_pred_cond = model.forward(
                    x=latents,
                    t=timestep,
                    context=positive_embeds,
                    context_mask=positive_mask,
                    img_shapes=img_shapes,
                    txt_seq_lens=positive_txt_seq_lens,
                )

                if use_cfg:
                    noise_pred_uncond = model.forward(
                        x=latents,
                        t=timestep,
                        context=negative_embeds,
                        context_mask=negative_mask,
                        img_shapes=img_shapes,
                        txt_seq_lens=negative_txt_seq_lens,
                    )

                    combined = noise_pred_uncond + float(cfg_scale) * (noise_pred_cond - noise_pred_uncond)
                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                    noise_norm = torch.norm(combined, dim=-1, keepdim=True)

                    noise_pred = combined * (cond_norm / (noise_norm + 1e-8))
                else:
                    noise_pred = noise_pred_cond

                prev_latents = sample_scheduler.step(noise_pred, t, latents)
                latents = prev_latents.to(dtype=run_dtype)

                if comfy_bar_callback is not None:
                    comfy_bar_callback(i + 1, total_steps)

        if offload_device is not None and offload_device != device:
            model.to(offload_device)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        latents = self._unpack_latents(latents, height, width, self.default_settings.vae.scale_factor)
        return latents
