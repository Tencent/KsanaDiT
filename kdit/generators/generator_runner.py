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

"""GeneratorRunner — Plan E 的核心执行引擎。

主流程精确移植自旧架构的 run() / for_batches() / run_one_batch()，
将所有 self.xxx() hook 调用替换为 Handler 调用。
GeneratorRunner 是 final 类，没有子类，没有 @abstractmethod。
所有模型差异通过 self._text / self._latent / self._denoise 三个 Handler 注入。
"""

import torch
from tqdm import tqdm

from kdit.models import KsanaDiffusionModel
from kdit.sample_solvers import get_sample_scheduler
from kdit.scheduler import KsanaBatchScheduler
from kdit.utils import log
from kdit.utils.profile import time_profile

from .generator_context import GeneratorInferContext
from .generator_def import GeneratorDef
from .steps import noise as noise_ops
from .steps import tensor_ops, validation


class GeneratorRunner:
    """Plan E 核心执行引擎 — 所有模型共享同一主流程。

    模型差异通过 GeneratorDef 中的三个 Handler 注入：
    - _text:    TextHandler    — 文本 conditioning 处理
    - _latent:  LatentHandler  — latent 预处理 / pack / unpack
    - _denoise: DenoiseHandler — 去噪循环钩子
    """

    def __init__(self, generator_def: GeneratorDef):
        self.gdef = generator_def
        self.model_key = generator_def.model_key
        self.batch_scheduler = KsanaBatchScheduler()
        self._text = generator_def.text_handler
        self._latent = generator_def.latent_handler
        self._denoise = generator_def.denoise_handler

    # ------------------------------------------------------------------
    # run() — 主入口
    # ------------------------------------------------------------------

    @time_profile
    def run(self, ctx: GeneratorInferContext) -> torch.Tensor:
        """执行完整的去噪生成流程。

        Args:
            ctx: 结构化输入上下文，包含模型、tensor、设备、配置等全部参数。
        Returns:
            latents (torch.Tensor)
        """
        # 解包 ctx
        diffusion_model = ctx.diffusion_model
        positive = ctx.positive
        negative = ctx.negative
        base_latent_obj = ctx.base_latent  # BaseLatent — 必须存在
        aux_latent_obj = ctx.aux_latent  # AuxLatent | None
        device = ctx.device
        offload_device = ctx.offload_device
        sample_config = ctx.sample_config
        runtime_config = ctx.runtime_config
        cache_config = ctx.cache_config
        video_control = ctx.video_control
        control_video_config = ctx.control_video_config
        comfy_bar_callback = ctx.comfy_bar_callback

        # base_latent 现在是必须的 — noise_shape 从 base_latent.latent.shape[1:] 推导
        if base_latent_obj is None:
            raise ValueError("base_latent is required — use VAE_COMPUTE_SHAPE or VAE_ENCODE_SPATIAL to create it")
        noise_shape = list(base_latent_obj.latent.shape[1:])

        # 从 BaseLatent/AuxLatent 对象中提取原始 tensor
        base_latent_list = [base_latent_obj.latent]
        if base_latent_obj.mask is not None:
            base_latent_list.append(base_latent_obj.mask)
        aux_latent = aux_latent_obj.latent if aux_latent_obj is not None else None

        diffusion_model = validation.valid_diffusion_model(diffusion_model, self.model_key)
        positive = self._text.preprocess(positive)
        negative = self._text.preprocess(negative)
        base_latent = self._latent.preprocess_base(base_latent_list)
        num_prompts = self._text.get_num_prompts(positive)

        sample_config = validation.valid_sample_config(sample_config, len(diffusion_model))
        cache_config = validation.valid_cache_config(cache_config, len(diffusion_model))
        runtime_config = validation.valid_runtime_config(runtime_config, num_prompts)
        positive, negative = self._text.validate(positive, negative)

        noise_shape = self._latent.validate_noise_shape(noise_shape, diffusion_model, self.model_key)

        self._apply_rope_function_to_models(diffusion_model, runtime_config.rope_function)

        # expand base_latent, aux_latent, positive and negative to total batch size supporting batch_size_per_prompts
        base_latent_expanded = self._valid_base_latent_to_total_prompts_size(
            base_latent, num_prompts, runtime_config.batch_size_per_prompts
        )
        aux_latent_expanded = self._valid_aux_latent_to_total_prompts_size(
            aux_latent, num_prompts, runtime_config.batch_size_per_prompts
        )
        positive, negative = self._text.expand_to_batch(positive, negative, runtime_config.batch_size_per_prompts)
        run_dtype = diffusion_model[0].run_dtype
        positive, negative = self._text.cast_to(positive, negative, dtype=run_dtype, device=device)
        base_latent_expanded = (
            tensor_ops.cast_to(base_latent_expanded, dtype=run_dtype, device=device)
            if base_latent_expanded is not None
            else None
        )
        aux_latent_expanded = self._cast_aux_latent(aux_latent_expanded, dtype=run_dtype, device=device)

        # create noise latents and batch strategy
        total_samples_num = sum(runtime_config.batch_size_per_prompts)
        noise_latents, seed_g = noise_ops.create_random_noise_latents(
            total_samples_num, noise_shape, runtime_config, device=device, dtype=run_dtype
        )
        batch_strategy = self.batch_scheduler.build_batch_strategy(
            self.model_key, noise_latents.shape, total_samples_num, run_dtype, device
        )
        validation.valid_aux_latent(aux_latent, noise_latents.shape)
        # Note: pack need after build strategy since strategy use noise_latents shape as 5D tensor
        patch_size = self._get_patch_size(diffusion_model)
        noise_latents = self._latent.pack_noise(noise_latents, patch_size)

        if aux_latent_expanded is not None:
            aux_latent_expanded = self._latent.pack_aux(aux_latent_expanded, patch_size)

        log.info(
            f"num_prompts: {num_prompts}, batch_size_per_prompts: {runtime_config.batch_size_per_prompts}, "
            f"total_samples_num: {total_samples_num} split as {len(batch_strategy)} batches"
        )

        video_control_kwargs = self._denoise.build_video_control_kwargs(
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
        noise_latents = self._for_batches(
            batch_strategy,
            diffusion_model=diffusion_model,
            noise_latents=noise_latents,
            positive=positive,
            negative=negative,
            base_latent=base_latent_expanded,
            sample_config=sample_config,
            aux_latent=aux_latent_expanded,
            run_steps_kwargs=run_steps_kwargs,
        )
        noise_latents = self._latent.unpack_noise(noise_latents, patch_size)
        noise_latents = self._denoise.post_run(noise_latents, video_control_kwargs)

        # TODO(qian): add auto estimate memory for automatic loading for all models
        if offload_device:
            [model.to(offload_device) for model in diffusion_model]

        # Note: [total_samples_num, vae_z_dim, f, h, w] or [total_samples_num, vae_z_dim, h, w]
        return noise_latents

    # ------------------------------------------------------------------
    # _for_batches() — 批次循环
    # ------------------------------------------------------------------

    def _for_batches(
        self,
        batch_strategy,
        *,
        diffusion_model: list[KsanaDiffusionModel],
        noise_latents: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        base_latent: torch.Tensor | None,
        sample_config,
        aux_latent: torch.Tensor | None,
        run_steps_kwargs: dict,
    ):
        log.info(f"batch_strategy={batch_strategy}")
        num_batches = len(batch_strategy)
        default_settings = diffusion_model[0].default_settings
        sample_config = self._latent.maybe_update_sample_config(sample_config, noise_latents.shape, default_settings)

        num_train_timesteps = self._get_num_train_timesteps(default_settings)
        for batch_idx, strategy_item in enumerate(batch_strategy):
            log.info(
                f"batch_idx {batch_idx}(num_batches {num_batches}), "
                f"strategy start={strategy_item.start}, end={strategy_item.end}, "
                f"combine_cond_uncond={strategy_item.combine_cond_uncond}"
            )
            batch_positive = tensor_ops.split_tensors(positive, strategy_item.start, strategy_item.end)
            batch_negative = tensor_ops.split_tensors(negative, strategy_item.start, strategy_item.end)
            batch_noise_latent = tensor_ops.split_tensors(noise_latents, strategy_item.start, strategy_item.end)
            batch_base_latent = tensor_ops.split_tensors(base_latent, strategy_item.start, strategy_item.end)
            batch_aux_latent = tensor_ops.split_tensors(aux_latent, strategy_item.start, strategy_item.end)

            device = run_steps_kwargs["device"]
            sample_scheduler, _, timesteps = get_sample_scheduler(
                num_train_timesteps=num_train_timesteps, sample_config=sample_config, device=device
            )
            batch_noise_latent = self._latent.apply_aux_latent(
                batch_noise_latent, batch_aux_latent, sample_config, timesteps, num_train_timesteps
            )
            with torch.no_grad():
                processed_latents = self._run_one_batch(
                    process_info=[batch_idx, num_batches],
                    diffusion_model=diffusion_model,
                    sample_config=sample_config,
                    positive=batch_positive,
                    negative=batch_negative,
                    noise_latent=batch_noise_latent,
                    base_latent=batch_base_latent,
                    timesteps=timesteps,
                    sample_scheduler_step_func=sample_scheduler.step,
                    sample_scheduler=sample_scheduler,  # Pass full scheduler for bidirectional sampling
                    combine_cond_uncond=strategy_item.combine_cond_uncond,
                    aux_latent=batch_aux_latent,
                    **run_steps_kwargs,
                )

            noise_latents[strategy_item.start : strategy_item.end] = processed_latents
        return noise_latents

    # ------------------------------------------------------------------
    # _run_one_batch() — 单批次去噪循环
    # ------------------------------------------------------------------

    def _run_one_batch(
        self,
        diffusion_model,
        positive,
        negative,
        noise_latent,
        base_latent,
        process_info,
        sample_config,
        runtime_config,
        cache_config,
        combine_cond_uncond,
        timesteps,
        run_dtype,
        sample_scheduler_step_func,
        sample_scheduler,
        seed_g,
        device,
        offload_device=None,
        comfy_bar_callback=None,
        video_control_kwargs=None,
        aux_latent=None,
    ) -> torch.Tensor:
        log.info(f"timesteps:{timesteps}, combine_cond_uncond:{combine_cond_uncond}")
        dit_cache = noise_ops.create_cache(cache_config, self.model_key)
        denoise_video_control_args = self._denoise.init_denoising_loop(
            video_control_kwargs, diffusion_model, sample_scheduler
        )

        total_steps = len(timesteps)
        cur_batch_size = self._text.get_num_prompts(positive)
        for iter_id, t in enumerate(tqdm(timesteps)):
            with time_profile(f"step_{iter_id}", note=f"t={t.item():.4f}"):
                current_step_percent = iter_id / max(total_steps - 1, 1)
                noise_latent = noise_latent.to(run_dtype)
                timestep = t.repeat(cur_batch_size)
                timestep_id = t.item()
                running_model = self._denoise.get_running_model(
                    diffusion_model, timestep_id=timestep_id, device=device, offload_device=offload_device
                )
                if running_model.device != device:
                    running_model.to(device)
                running_cache = self._denoise.get_running_cache(dit_cache, timestep_id=timestep_id)
                running_cfg_scale = self._denoise.get_running_cfg_scale(
                    cfg_scale=sample_config.cfg_scale, timestep_id=timestep_id
                )

                step_kwargs = self._denoise.get_step_kwargs(
                    denoise_video_control_args, current_step_percent, iter_id, total_steps
                )

                forward_kargs = self._denoise.prepare_model_forward_kargs(
                    running_cfg_scale,
                    noise_latent=noise_latent,
                    timestep=timestep,
                    combine_cond_uncond=combine_cond_uncond,
                    step_iter=iter_id,
                    cache=running_cache,
                    positive=positive,
                    negative=negative,
                    base_latent=base_latent,
                    aux_latent=aux_latent,
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
                    noise_pred = self._denoise.apply_cfg(
                        running_cfg_scale,
                        noise_pred_cond,
                        noise_pred_uncond,
                        denoise_video_control_args=denoise_video_control_args,
                        step_index=iter_id,
                        total_steps=total_steps,
                    )
                else:
                    noise_pred = running_model.forward(**forward_kargs)

                noise_pred = self._denoise.post_noise_prediction(
                    noise_pred, noise_latent, t, denoise_video_control_args
                )

                noise_latent_shape = noise_latent.shape
                step_out = sample_scheduler_step_func(noise_pred, t, noise_latent, return_dict=False, generator=seed_g)
                noise_latent_forward = step_out[0] if isinstance(step_out, (tuple, list)) else step_out
                if noise_latent_forward.numel() != int(torch.prod(torch.tensor(noise_latent_shape))):
                    raise RuntimeError(
                        f"can not reshape {noise_latent_forward.shape} to {noise_latent_shape}, "
                        f" please debug sample solver"
                    )
                noise_latent_forward = noise_latent_forward.reshape(noise_latent_shape)

                noise_latent = self._denoise.finalize_step(
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
                    base_latent=base_latent,
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

    # ------------------------------------------------------------------
    # 内部工具方法（不通过 Handler）
    # ------------------------------------------------------------------

    @staticmethod
    def _use_cfg(cfg_scale: float, eps: float = 1e-6):
        return abs(cfg_scale - 1.0) > eps

    def _get_num_train_timesteps(self, default_settings):
        num_train_timesteps = getattr(default_settings.sample_config, "num_train_timesteps", None)
        if num_train_timesteps is None:
            raise RuntimeError("num_train_timesteps should be set in yaml sample_config settings")
        return num_train_timesteps

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

    def _apply_rope_function_to_models(self, diffusion_models: list[KsanaDiffusionModel], rope_function: str | None):
        rope_value = rope_function or "default"
        for model in diffusion_models:
            if hasattr(model.model, "set_rope_function"):
                model.model.set_rope_function(rope_value)

    def _valid_base_latent_to_total_prompts_size(
        self,
        base_latent: torch.Tensor | None,
        num_prompts: int,
        batch_size_per_prompts: list[int],
    ) -> torch.Tensor | None:
        """按 batch_size_per_prompts 扩展 base_latent tensor。"""
        if base_latent is None:
            return None
        expanded = self._expand_single_latents(base_latent, num_prompts, batch_size_per_prompts)
        return expanded[0]

    def _valid_aux_latent_to_total_prompts_size(
        self,
        aux_latent,
        num_prompts: int,
        batch_size_per_prompts: list[int],
    ):
        """按 batch_size_per_prompts 扩展 aux_latent，返回统一的 list[Tensor]。

        通过 isinstance 自动分发：
        - Tensor: shape[0] 是 batch 维度 → _expand_single_latents
        - list[Tensor]: list 长度 = prompt 数 → _expand_list_latents
        """
        if aux_latent is None:
            return None
        if isinstance(aux_latent, torch.Tensor):
            return self._expand_single_latents(aux_latent, num_prompts, batch_size_per_prompts)
        elif isinstance(aux_latent, list):
            return self._expand_list_latents(aux_latent, num_prompts, batch_size_per_prompts)
        else:
            raise TypeError(f"aux_latent must be Tensor, list[Tensor] or None, got {type(aux_latent)}")

    def _expand_list_latents(self, latents: list[torch.Tensor], num_prompts: int, batch_size_per_prompts: list[int]):
        """将 list[Tensor] 格式的 latents 按 batch_size_per_prompts 扩展。"""
        if len(latents) != num_prompts:
            raise ValueError(f"latents list length ({len(latents)}) must match num_prompts ({num_prompts})")
        stacked = torch.stack(latents)  # [num_prompts, num_refs, C, 1, H, W]
        expanded = self._text._expand_to_total_prompts_size(
            stacked, batch_size_per_prompts
        )  # [total_batch, num_refs, C, 1, H, W]
        # 转置结构：[total_batch, num_refs, ...] -> list[num_refs] of [total_batch, ...]
        return list(expanded.transpose(0, 1))  # list[num_refs] of [total_batch, C, 1, H, W]

    def _expand_single_latents(self, latent: torch.Tensor, num_prompts: int, batch_size_per_prompts: list[int]):
        """处理单个 tensor 格式的 latents。"""
        if num_prompts > latent.shape[0]:
            current_batch_size = latent.shape[0]
            repeats_num = num_prompts // current_batch_size
            if repeats_num * current_batch_size != num_prompts:
                raise ValueError(f"Cannot evenly distribute {current_batch_size} images to {num_prompts} prompts")
            latent = latent.repeat_interleave(repeats_num, dim=0)
        return [self._text._expand_to_total_prompts_size(latent, batch_size_per_prompts)]

    def _cast_aux_latent(
        self,
        aux_latent: list[torch.Tensor] | torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor] | torch.Tensor | None:
        """将 aux_latent 转换到目标 dtype 和 device，支持 Tensor 和 list[Tensor]。"""
        if aux_latent is None:
            return None
        if isinstance(aux_latent, torch.Tensor):
            return tensor_ops.cast_to(aux_latent, dtype=dtype, device=device)
        if isinstance(aux_latent, list):
            return [tensor_ops.cast_to(t, dtype=dtype, device=device) for t in aux_latent]
        return aux_latent
