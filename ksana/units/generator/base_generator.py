# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import sys
from abc import abstractmethod

import torch
from tqdm import tqdm

from ksana.cache import create_hybrid_cache
from ksana.config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from ksana.config.cache_config import KsanaCacheConfig, KsanaHybridCacheConfig, warp_as_hybrid_cache
from ksana.config.video_control_config import KsanaVideoControlConfig
from ksana.models import KsanaDiffusionModel, KsanaModelKey
from ksana.sample_solvers import get_sample_scheduler
from ksana.scheduler import KsanaBatchScheduler
from ksana.utils import evolve_with_recommend, log, time_range
from ksana.utils.vace import KsanaVaceContext

from ..runner_unit import KsanaRunnerUnit


class KsanaBaseGenerator(KsanaRunnerUnit):
    def __init__(self):
        super().__init__()
        self.batch_scheduler = KsanaBatchScheduler()

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
            if len(sample_config.cfg_scale) < model_len:
                raise ValueError(f"cfg_scale length must be {model_len}, but got {len(sample_config.cfg_scale)}")
            evolve_with_recommend(sample_config, {"cfg_scale": list(sample_config.cfg_scale)}, force_update=True)
        else:
            raise TypeError(f"sample_config.cfg_scale {sample_config.cfg_scale} type not supported")
        if sample_config.solver is None or not KsanaSolverType.support(sample_config.solver):
            raise ValueError(f"sample_config.solver must in support list {KsanaSolverType.get_supported_list()}")
        if sample_config.denoise <= 0.0:
            raise ValueError(f"denoise <= 0.0 is not supported, got {sample_config.denoise}")
        return sample_config

    def _valid_cache_config(
        self, cache_config: KsanaCacheConfig | KsanaHybridCacheConfig, model_len: int
    ) -> KsanaHybridCacheConfig:
        log.info(f"cache_config: {cache_config}")
        if cache_config is None:
            return
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
        self, image_embeds: list[torch.Tensor] | None, num_prompts: int, batch_size_per_prompts: list[int]
    ):
        if image_embeds is None:
            return None
        if len(image_embeds) == 1:
            return self._expand_single_latents(image_embeds[0], num_prompts, batch_size_per_prompts)
        else:
            return self._expand_list_latents(image_embeds, num_prompts, batch_size_per_prompts)

    def _expand_list_latents(
        self, image_embeds: list[torch.Tensor], num_prompts: int, batch_size_per_prompts: list[int]
    ):
        """将 list[Tensor] 格式的 image_embeds 按 batch_size_per_prompts 扩展"""
        if len(image_embeds) != num_prompts:
            raise ValueError(f"image_embeds list length ({len(image_embeds)}) must match num_prompts ({num_prompts})")
        stacked = torch.stack(image_embeds)  # [num_prompts, num_refs, C, 1, H, W]
        expanded = self._expand_to_total_prompts_size(
            stacked, batch_size_per_prompts
        )  # [total_batch, num_refs, C, 1, H, W]
        # 转置结构：[total_batch, num_refs, ...] -> list[num_refs] of [total_batch, ...]
        return list(expanded.transpose(0, 1))  # list[num_refs] of [total_batch, C, 1, H, W]

    def _expand_single_latents(self, image_embeds: torch.Tensor, num_prompts: int, batch_size_per_prompts: list[int]):
        """处理单个tensor格式的latents"""
        # 如果需要重复latents以匹配prompt数量
        if num_prompts > image_embeds.shape[0]:
            current_batch_size = image_embeds.shape[0]
            repeats_num = num_prompts // current_batch_size
            if repeats_num * current_batch_size != num_prompts:
                raise ValueError(f"Cannot evenly distribute {current_batch_size} images to {num_prompts} prompts")
            image_embeds = image_embeds.repeat_interleave(repeats_num, dim=0)
        return [self._expand_to_total_prompts_size(image_embeds, batch_size_per_prompts)]

    def _valid_prompts_to_total_prompts_size(
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

    def _valid_input_latent(self, input_latent: torch.Tensor, noise_shape: tuple[int]):
        if input_latent is None:
            return
        if input_latent.dim() != len(noise_shape) or len(noise_shape) != 5:  # [bs, z_dim, f, h, w]
            raise ValueError(
                f"input_latent.dim() {input_latent.dim()} must be equal to noise_shape.len()"
                f" {len(noise_shape)} and both must be 5"
            )
        input_bs, input_z_dim, _, input_h, input_w = input_latent.shape
        noise_bs, noise_z_dim, _, noise_h, noise_w = noise_shape
        if input_bs != noise_bs or input_z_dim != noise_z_dim or input_h != noise_h or input_w != noise_w:
            raise ValueError(
                f"input_latent shape {input_latent.shape} must match "
                f" noise_shape {noise_shape} in all dimensions except frame dimension"
            )

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

    def _apply_input_latent(self, *args, **kwargs):
        raise NotImplementedError("subclass must implement _apply_input_latent method")

    def run_one_batch(
        self,
        diffusion_model: list[KsanaDiffusionModel],
        positive: torch.Tensor | tuple,
        negative: torch.Tensor | tuple,
        noise_latent: torch.Tensor,
        image_embeds: list[torch.Tensor] | None,
        process_info: list[int],
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig],
        combine_cond_uncond: bool,
        timesteps: torch.Tensor,  # Tensor(list[int])
        run_dtype: torch.dtype,
        sample_scheduler_step_func,
        sample_scheduler,
        seed_g: torch.Generator,
        device: torch.device,
        offload_device: torch.device = None,
        comfy_bar_callback=None,
        video_control_kwargs: dict | None = None,
    ) -> torch.Tensor:
        log.info(f"timesteps:{timesteps}, combine_cond_uncond:{combine_cond_uncond}")
        dit_cache = self._create_cache(cache_config, self.model_key)
        denoise_video_control_args = self.init_denoising_loop(video_control_kwargs, diffusion_model, sample_scheduler)

        total_steps = len(timesteps)
        cur_batch_size = self._get_num_prompts(positive)
        for iter_id, t in enumerate(tqdm(timesteps)):
            current_step_percent = iter_id / max(total_steps - 1, 1)
            noise_latent = noise_latent.to(run_dtype)
            timestep = t.repeat(cur_batch_size)
            timestep_id = t.item()
            running_model = self.get_running_model(
                diffusion_model, timestep_id=timestep_id, device=device, offload_device=offload_device
            )
            if running_model.device != device:
                running_model.to(device)
            running_cache = self.get_running_cache(dit_cache, timestep_id=timestep_id)
            running_cfg_scale = self.get_running_cfg_scale(cfg_scale=sample_config.cfg_scale, timestep_id=timestep_id)

            step_kwargs = self.get_step_kwargs(denoise_video_control_args, current_step_percent, iter_id, total_steps)

            forward_kargs = self.prepare_model_forward_kargs(
                running_cfg_scale,
                noise_latent=noise_latent,
                timestep=timestep,
                combine_cond_uncond=combine_cond_uncond,
                step_iter=iter_id,
                cache=running_cache,
                positive=positive,
                negative=negative,
                image_embeds=image_embeds,
                **step_kwargs,
            )
            if self._use_cfg(running_cfg_scale):
                if combine_cond_uncond:
                    noise_pred_batch = running_model.forward(**forward_kargs)
                    noise_pred_cond, noise_pred_uncond = noise_pred_batch.chunk(2, dim=0)
                else:
                    if not isinstance(forward_kargs, (tuple, list)) or len(forward_kargs) != 2:
                        raise ValueError(f"forward_kargs {forward_kargs} must be tuple of (arg_cond, arg_uncond)")
                    arg_cond, arg_uncond = forward_kargs
                    noise_pred_cond = running_model.forward(**arg_cond)
                    noise_pred_uncond = running_model.forward(**arg_uncond)
                noise_pred = self.apply_cfg(
                    running_cfg_scale,
                    noise_pred_cond,
                    noise_pred_uncond,
                    denoise_video_control_args=denoise_video_control_args,
                    step_index=iter_id,
                    total_steps=total_steps,
                )
            else:
                noise_pred = running_model.forward(**forward_kargs)

            noise_pred = self.post_noise_prediction(noise_pred, noise_latent, t, denoise_video_control_args)

            noise_latent_shape = noise_latent.shape
            step_out = sample_scheduler_step_func(noise_pred, t, noise_latent, return_dict=False, generator=seed_g)
            noise_latent_forward = step_out[0] if isinstance(step_out, (tuple, list)) else step_out
            if noise_latent_forward.numel() != int(torch.prod(torch.tensor(noise_latent_shape))):
                raise RuntimeError(
                    f"can not reshape {noise_latent_forward.shape} to {noise_latent_shape}, please debug sample solver"
                )
            noise_latent_forward = noise_latent_forward.reshape(noise_latent_shape)

            noise_latent = self.finalize_step(
                noise_latent,
                noise_latent_forward,
                denoise_video_control_args,
                running_model=running_model,
                running_cfg_scale=running_cfg_scale,
                timestep=timestep,
                t=t,
                iter_id=iter_id,
                total_steps=total_steps,
                current_step_percent=current_step_percent,
                combine_cond_uncond=combine_cond_uncond,
                positive=positive,
                negative=negative,
                image_embeds=image_embeds,
                step_kwargs=step_kwargs,
                sample_config=sample_config,
                seed_g=seed_g,
            )

            if comfy_bar_callback is not None:
                steps = sample_config.steps
                batch_idx, num_batches = process_info
                current_step_iter = batch_idx * steps + (iter_id + 1)
                comfy_bar_callback(current_step_iter, num_batches * steps)
        if dit_cache is not None:
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
        image_embeds: list[torch.Tensor] | None,
        sample_config: KsanaRuntimeConfig,
        input_latent: torch.Tensor,
        run_steps_kwargs: dict,
    ):
        log.info(f"batch_strategy={batch_strategy}")
        num_batches = len(batch_strategy)
        default_settings = diffusion_model[0].default_settings
        sample_config = self.maybe_update_sample_config(sample_config, noise_latents.shape, default_settings)

        num_train_timesteps = self._get_num_train_timesteps(default_settings)
        for batch_idx, strategy_item in enumerate(batch_strategy):
            log.info(
                f"batch_idx {batch_idx}(num_batches {num_batches}), "
                f"strategy start={strategy_item.start}, end={strategy_item.end}, "
                f"combine_cond_uncond={strategy_item.combine_cond_uncond}"
            )
            batch_positive = self._split_tensors(positive, strategy_item.start, strategy_item.end)
            batch_negative = self._split_tensors(negative, strategy_item.start, strategy_item.end)
            batch_noise_latent = self._split_tensors(noise_latents, strategy_item.start, strategy_item.end)
            batch_image_embeds = self._split_tensors(image_embeds, strategy_item.start, strategy_item.end)
            batch_input_latent = self._split_tensors(input_latent, strategy_item.start, strategy_item.end)

            device = run_steps_kwargs["device"]
            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=num_train_timesteps, sample_config=sample_config, device=device
            )
            batch_noise_latent = self._apply_input_latent(
                batch_noise_latent, batch_input_latent, sample_config, timesteps, num_train_timesteps
            )
            with torch.no_grad():
                processed_latents = self.run_one_batch(
                    process_info=[batch_idx, num_batches],
                    diffusion_model=diffusion_model,
                    sample_config=sample_config,
                    positive=batch_positive,
                    negative=batch_negative,
                    noise_latent=batch_noise_latent,
                    image_embeds=batch_image_embeds,
                    timesteps=timesteps,
                    sample_scheduler_step_func=sample_scheduler.step,
                    sample_scheduler=sample_scheduler,  # Pass full scheduler for bidirectional sampling
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
        image_embeds: list[torch.Tensor] | None = None,
        input_latent: torch.Tensor = None,
        cache_config: list[KsanaCacheConfig | KsanaHybridCacheConfig] = None,
        device=None,
        offload_device=None,
        comfy_bar_callback=None,
        video_control: KsanaVideoControlConfig | None = None,
        control_video_config: KsanaVaceContext | None = None,
    ) -> torch.Tensor:
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_config (KsanaSampleConfig): _description_
        Returns:
            latents (torch.Tensor)
        """
        diffusion_model = self._valid_diffusion_model(diffusion_model)
        positive = self.preprocess_text_conditioning(positive)
        negative = self.preprocess_text_conditioning(negative)
        image_embeds = self.preprocess_image_embeds(image_embeds)
        num_prompts = self._get_num_prompts(positive)

        sample_config = self._valid_sample_config(sample_config, len(diffusion_model))
        cache_config = self._valid_cache_config(cache_config, len(diffusion_model))
        runtime_config = self._valid_runtime_config(runtime_config, num_prompts)
        positive, negative = self._valid_prompts(positive, negative)

        noise_shape = self.valid_noise_shape(noise_shape, diffusion_model)

        self._apply_rope_function_to_models(diffusion_model, runtime_config.rope_function)

        # expand image_embeds, positive and negative to total batch size supporting batch_size_per_prompts
        image_embeds = self._valid_image_to_total_prompts_size(
            image_embeds, num_prompts, runtime_config.batch_size_per_prompts
        )
        positive, negative = self._valid_prompts_to_total_prompts_size(
            positive, negative, runtime_config.batch_size_per_prompts
        )
        run_dtype = diffusion_model[0].run_dtype
        positive, negative = self.cast_text_tensors_to(positive, negative, dtype=run_dtype, device=device)
        image_embeds = self.cast_image_tensor_to(image_embeds, dtype=run_dtype, device=device)

        # create noise latents and batch strategy
        total_samples_num = sum(runtime_config.batch_size_per_prompts)
        noise_latents, seed_g = self._create_random_noise_latents(
            total_samples_num, noise_shape, runtime_config, device=device, dtype=run_dtype
        )
        batch_strategy = self.batch_scheduler.build_batch_strategy(
            self.model_key, noise_latents.shape, total_samples_num, run_dtype, device
        )
        self._valid_input_latent(input_latent, noise_latents.shape)
        # Note: pack need after build strategy since strategy use noise_latents shape as 5D tensor
        patch_size = self._get_patch_size(diffusion_model)
        noise_latents = self.pack_noise_latents(noise_latents, patch_size)

        if image_embeds is not None and len(image_embeds) > 0:
            image_embeds = self.pack_ref_latents(image_embeds, patch_size)

        log.info(
            f"num_prompts: {num_prompts}, batch_size_per_prompts: {runtime_config.batch_size_per_prompts}, "
            f"total_samples_num: {total_samples_num} split as {len(batch_strategy)} batches"
        )

        video_control_kwargs = self.build_video_control_kwargs(
            control_video_config=control_video_config,
            device=device,
            sample_config=sample_config,
            video_control=video_control,
        )

        run_steps_kwargs = {
            "cache_config": cache_config,
            "runtime_config": runtime_config,
            "seed_g": seed_g,
            "run_dtype": run_dtype,
            "device": device,
            "offload_device": offload_device,
            "comfy_bar_callback": comfy_bar_callback,
            "video_control_kwargs": video_control_kwargs,
        }
        noise_latents = self.for_batches(
            batch_strategy,
            diffusion_model=diffusion_model,
            noise_latents=noise_latents,
            positive=positive,
            negative=negative,
            image_embeds=image_embeds,
            sample_config=sample_config,
            input_latent=input_latent,
            run_steps_kwargs=run_steps_kwargs,
        )
        noise_latents = self.unpack_noise_latents(noise_latents, patch_size)
        noise_latents = self.post_run(noise_latents, video_control_kwargs)

        # TODO(qian): add auto estimate memory for automatic loading for all models
        if offload_device:
            [model.to(offload_device) for model in diffusion_model]

        # Note: [total_samples_num, vae_z_dim, f, h, w] or [total_samples_num, vae_z_dim, h, w]
        return noise_latents

    def _split_tensors(self, tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor], start: int, end: int):
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
    def preprocess_image_embeds(self, image_embeds):
        return image_embeds

    def preprocess_text_conditioning(self, text_conditioning: torch.Tensor | tuple):
        return text_conditioning

    def valid_noise_shape(self, noise_shape: tuple[int] | list[int], diffusion_model):
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

    def _get_patch_size(self, diffusion_model: list[KsanaDiffusionModel]):
        model = diffusion_model[0] if isinstance(diffusion_model, (list, tuple)) else diffusion_model
        default_settings = model.default_settings
        patch_size = getattr(default_settings.diffusion, "patch_size", None)
        patch_size = getattr(diffusion_model, "patch_size", None) or patch_size
        if patch_size is None:
            raise RuntimeError(
                f"{self.model_key} can not get patch_size from diffusion_model or default_settings, "
                "should patch_size add to default_settings.diffusion"
            )
        log.info(f"{self.model_key} patch_size: {patch_size}")
        return patch_size

    def pack_noise_latents(self, noise_latents: torch.Tensor, patch_size) -> torch.Tensor:
        return noise_latents

    def pack_ref_latents(self, ref_latents: list[torch.Tensor], patch_size: int) -> list[torch.Tensor]:
        return ref_latents

    def unpack_noise_latents(self, noise_latents: torch.Tensor, patch_size) -> torch.Tensor:
        return noise_latents

    def maybe_update_sample_config(self, sample_config, *_):
        return sample_config

    def cast_text_tensors_to(self, positive, negative, *, dtype: torch.dtype, device: torch.device):
        positive = self._cast_to(positive, dtype=dtype, device=device)
        negative = self._cast_to(negative, dtype=dtype, device=device)
        return positive, negative

    def cast_image_tensor_to(
        self, image_embeds: list[torch.Tensor] | None, *, dtype: torch.dtype, device: torch.device
    ):
        if image_embeds is None:
            return None
        return [self._cast_to(embed, dtype=dtype, device=device) for embed in image_embeds]

    def get_running_cfg_scale(self, cfg_scale: list[float], **kwargs):
        return cfg_scale[0] if isinstance(cfg_scale, (list, tuple)) else cfg_scale

    def get_running_model(self, diffusion_model: list[KsanaDiffusionModel], **kwargs):
        return diffusion_model[0] if isinstance(diffusion_model, (list, tuple)) else diffusion_model

    def get_running_cache(self, dit_cache: list, **kwargs):
        return dit_cache[0] if isinstance(dit_cache, (list, tuple)) else dit_cache

    def apply_cfg(self, cfg_scale, cond, uncond, **kwargs):
        return uncond + float(cfg_scale) * (cond - uncond)

    def init_denoising_loop(self, video_control_kwargs, diffusion_model, sample_scheduler):
        return {}

    def get_step_kwargs(self, denoise_video_control_args, current_step_percent, iter_id, total_steps):
        return {}

    def post_noise_prediction(self, noise_pred, noise_latent, t, denoise_video_control_args):
        return noise_pred

    def finalize_step(self, noise_latent, noise_latent_forward, denoise_video_control_args, **step_state):
        return noise_latent_forward

    def build_video_control_kwargs(self, **kwargs):
        return {}

    def post_run(self, noise_latents, video_control_kwargs):
        return noise_latents

    @abstractmethod
    def prepare_model_forward_kargs(self, cfg_scale, **kwargs) -> dict | tuple[dict, dict]:
        raise NotImplementedError("prepare_model_forward_kargs must be implemented in subclass")
