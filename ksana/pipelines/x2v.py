from abc import ABC, abstractmethod
import torch
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import math

import random
import sys
from dataclasses import dataclass, field
from ..config import KsanaSampleConfig, KsanaRuntimeConfig, KsanaPipelineConfig, KsanaModelConfig
from ..utils import log, print_recursive, time_range, MemoryProfiler, is_dir
from ..cache import create_cache
from ..sample_solvers import get_sample_scheduler
from ..scheduler import KsanaScheduler

from ..models.model_key import KsanaModelKey
from ..models.model_pool import KsanaModelPool
from ..models.base_model import KsanaModel


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
        self.scheduler = KsanaScheduler(pipeline_config)

        # Note: only save model_key, do NOT pass model itself
        self.text_encoder_key = None
        self.vae_key = None
        self.diffusion_model_keys = None
        self.has_lora = False

    @abstractmethod
    def load_text_encoder(self, checkpoint_dir, shard_fn=None) -> tuple[KsanaModelKey, KsanaModel]:
        pass

    @abstractmethod
    def load_vae(self, checkpoint_dir, device) -> tuple[KsanaModelKey, KsanaModel]:
        pass

    def clear_models(self):
        self.text_encoder_key = None
        self.vae_key = None
        self.diffusion_model_keys = None
        self.has_lora = False

    @abstractmethod
    def load_diffusion_model(
        self,
        model_path,
        *,
        lora: None | str | list[list[dict], list[dict]] = None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        input_model_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ) -> list[tuple[KsanaModelKey, KsanaModel]]:
        pass

    def load_models(
        self,
        model_path,
        *,
        text_checkpoint_dir=None,
        vae_checkpoint_dir=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
        lora: None | str | list[list[dict], list[dict]] = None,
    ) -> list[tuple[KsanaModelKey, KsanaModel]]:
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
        self.has_lora = lora is not None

        self.text_encoder_key, text_encoder = self.load_text_encoder(text_checkpoint_dir, shard_fn=shard_fn)
        if offload_device:
            text_encoder.to(offload_device)
        diffusion_model_tuple_list = self.load_diffusion_model(
            model_path,
            lora=lora,
            model_config=model_config,
            dist_config=dist_config,
            device=device,
            offload_device=offload_device,
            shard_fn=shard_fn,
        )
        if offload_device:
            [diffusion_model.to(offload_device) for _, diffusion_model in diffusion_model_tuple_list]
        self.diffusion_model_keys = [one_model_key for one_model_key, _ in diffusion_model_tuple_list]

        self.vae_key, vae_model = self.load_vae(vae_checkpoint_dir, device)
        self.vae_z_dim = vae_model.z_dim
        self.vae_stride = vae_model.vae_stride
        if offload_device:
            vae_model.to(offload_device)
        return [(self.text_encoder_key, text_encoder), (self.vae_key, vae_model), *diffusion_model_tuple_list]

    def forward_text_encoder(
        self,
        model_pool: KsanaModelPool,
        prompts_positive,
        prompts_negative=None,
        device=None,
        offload_device=None,
        offload_model=False,
    ):
        text_encoder = model_pool.get_model(self.text_encoder_key)
        bs = len(prompts_positive)
        assert bs > 0, "prompts must not be empty"
        default_neg_prompt = self.pipeline_config.default_config.sample_neg_prompt
        prompts_negative = prompts_negative if prompts_negative is not None else [default_neg_prompt] * bs
        assert len(prompts_positive) == len(
            prompts_negative
        ), f"The number of negative prompts ({len(prompts_negative)}) must match the number of positive prompts ({bs})."

        assert device is not None
        if text_encoder.device != device:
            text_encoder.to(device)

        all_prompts = prompts_positive + prompts_negative
        all_embeddings_list = text_encoder.forward(all_prompts)

        # TODO(qiannan): self.text_encoder.forward tokenizer的时候是填充到相同长度了，但是返回是裁剪了，所以如果返回不裁剪，就不需要pad了
        # Pad the combined list of tensors to the max length in the entire batch.
        all_padded_embeddings = pad_sequence(all_embeddings_list, batch_first=True, padding_value=0.0)

        # Split the padded tensor back into positive and negative parts.
        positive, negative = torch.chunk(all_padded_embeddings, 2, dim=0)

        if offload_model and offload_device is not None and offload_device != device:
            text_encoder.to(offload_device)

        return positive, negative

    @abstractmethod
    def process_input_cache(self, cache_method):
        pass

    def forward_vae(
        self, model_pool: KsanaModelPool, latents, local_rank, device=None, offload_device=None, offload_model=False
    ):
        # TODO: support multi gpu
        if local_rank != 0:
            return
        vae_model = model_pool.get_model(self.vae_key)
        if vae_model.device != device:
            vae_model.to(device)
        videos = vae_model.decode(latents)
        if offload_model and offload_device is not None and offload_device != device:
            vae_model.to(offload_device)
        del latents
        log.info(f"Generated video count: {len(videos)}, first video shape: {videos[0].shape if videos else 'empty'}")
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
            high_cache.offload_to_cpu()
            return low_cache

    def _run_inference_loop(
        self,
        positive_batch: torch.Tensor,
        negative_batch: torch.Tensor,
        latents_batch: torch.Tensor,
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
        """运行推理循环处理单个batch"""

        high_sample_guide_scale, low_sample_guide_scale = guide_scales
        high_model, low_model = models
        high_cache_config, low_cache_config = cache_configs if cache_configs else (None, None)
        bar_info, comfy_bar_callback = bar_info_callback if bar_info_callback else (None, None)

        # 每次重新创建cache
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

        batch_size_current = positive_batch.shape[0]

        arg_cond = {"phase": "cond", "context": positive_batch, "seq_len": seq_len}
        arg_uncond = {"phase": "uncond", "context": negative_batch, "seq_len": seq_len}
        arg_batch = None
        if combine_cond_uncond:
            arg_batch = {
                "phase": "batch",
                "context": torch.cat([positive_batch, negative_batch], dim=0),
                "seq_len": seq_len,
            }

        total_steps = len(timesteps)
        for iter_id, t in enumerate(tqdm(timesteps)):
            MemoryProfiler.record_memory(f"before_inference_loop_iter_{iter_id}")

            latent_model_input = latents_batch.to(run_dtype)
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
                offload_device=offload_device,
            )
            run_model = run_model.to(device)

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
                    noise_pred_batch = run_model.forward(x=latent_batch, t=timestep_batch, cache=run_cache, **arg_batch)
                    # 分离结果: [2*bs, 16, fi, hi, wi] => 2 x [bs, 16, fi, hi, wi]
                    noise_pred_cond, noise_pred_uncond = noise_pred_batch.chunk(2, dim=0)
                else:
                    noise_pred_uncond = run_model.forward(
                        x=latent_model_input, t=timestep, cache=run_cache, **arg_uncond
                    )
                    noise_pred_cond = run_model.forward(x=latent_model_input, t=timestep, cache=run_cache, **arg_cond)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = run_model.forward(x=latent_model_input, t=timestep, cache=run_cache, **arg_cond)

            temp_x0 = sample_scheduler_step_func(
                noise_pred,
                t,
                latents_batch,
                return_dict=False,
                generator=seed_g,
            )
            latents_batch = temp_x0 if sample_config_solver_name == "euler" else temp_x0[0]
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

        # 显示cache统计信息
        if high_cache is not None:
            high_cache.show_cache_rate()
        if low_cache is not None:
            low_cache.show_cache_rate()

        return latents_batch

    def valid_args(self, diffusion_models: list[KsanaModel], sample_config):
        high_model = diffusion_models
        low_model = None
        if hasattr(diffusion_models, "__len__"):
            assert len(diffusion_models) == 2, f"size of model must be 2, but got {len(diffusion_models)}"
            high_model, low_model = diffusion_models
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
        assert sample_config.denoise > 0.0, f"denoise <= 0.0 is not supported, got {sample_config.denoise}"
        return (
            high_model,
            low_model,
            high_sample_guide_scale,
            low_sample_guide_scale,
        )

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
        self, target_shape: tuple[int], runtime_config: KsanaRuntimeConfig, device: torch.device, dtype: torch.dtype
    ):
        """
        Args:
            return tensor shape :[bs, z_dim, f, h, w] (5D tensor for batch)
        """
        seed = (
            runtime_config.seed
            if runtime_config.seed is not None and runtime_config.seed >= 0
            else random.randint(0, sys.maxsize)
        )
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        bs = target_shape[0]
        target_shape = target_shape[1:]
        latents_list = []
        for _ in range(bs):
            single_noise = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=device,
                generator=seed_g,
            ).to(dtype)
            latents_list.append(single_noise)
        noise = torch.stack(latents_list, dim=0)
        return noise, seed_g

    def get_seq_len(self, target_shape, patch_size: list[int], sp_size: int):
        _, _, f, h, w = target_shape
        return math.ceil((h * w) / (patch_size[1] * patch_size[2]) * f / sp_size) * sp_size

    @time_range
    def forward_diffusion_models_with_tensors(
        self,
        diffusion_models: list[KsanaModel],
        positive: torch.Tensor,  # [bs, 512, 4096]
        negative: torch.Tensor,  # [bs, 512, 4096]
        latents: torch.Tensor,  # [bs, 16, 2, 90, 160]
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        high_cache_config=None,
        low_cache_config=None,
        device=None,
        offload_device=None,
        comfy_bar_callback=None,
    ):
        """_summary_

        Args:
            positive (torch.Tensor): _description_
            sample_config (KsanaSampleConfig): _description_
        Returns:
            latents (torch.Tensor)
        """
        log.info(f"runtime_config: {runtime_config}, sample_config: {sample_config}")
        high_model, low_model, high_sample_guide_scale, low_sample_guide_scale = self.valid_args(
            diffusion_models, sample_config
        )
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
        batch_size = positive.shape[0]

        # TODO: consider image input
        if latents is not None:
            assert len(latents.shape) == 5, f"latents.shape {latents.shape} dim must be 5:(bs, z_dim:16, f, h, w)"
            target_shape = latents.shape
        else:
            size = runtime_config.size
            frame_num = runtime_config.frame_num
            # TODO: here should not used vae model params insider forward transformer, need create noise outside pipeline
            assert (
                self.vae_z_dim is not None and self.vae_stride is not None
            ), f"self.vae_z_dim {self.vae_z_dim}, self.vae_stride {self.vae_stride}"
            target_shape = (
                batch_size,
                self.vae_z_dim,
                (frame_num - 1) // self.vae_stride[0] + 1,
                size[1] // self.vae_stride[1],
                size[0] // self.vae_stride[2],
            )
            # => (z_dim:16, 2, 96, 160)

        latents, seed_g = self.create_random_noise_latents(target_shape, runtime_config, device, run_dtype)
        seq_len = self.get_seq_len(latents.shape, default_pipeline_config.patch_size, high_model.sp_size)

        latents = self.cast_to(latents, run_dtype, device)
        positive = self.cast_to(positive, run_dtype, device)
        negative = self.cast_to(negative, run_dtype, device)
        sample_scheduler_kwargs = dict(
            num_train_timesteps=default_pipeline_config.num_train_timesteps,
            sampling_steps=sample_config.steps,
            sample_solver=sample_config.solver,
            device=device,
            shift=sample_config.shift,
            denoise=sample_config.denoise,
        )

        with torch.no_grad():
            # 构建动态批处理策略
            batch_strategy = self.scheduler.build_batch_strategy(latents.shape, batch_size, run_dtype, device)

            # 计算全局进度信息
            total_steps_per_batch = sample_config.steps
            global_total_steps = len(batch_strategy) * total_steps_per_batch

            # 使用动态batch处理
            for batch_idx, strategy_item in enumerate(batch_strategy):
                pos_batch = positive[strategy_item.start : strategy_item.end]
                neg_batch = negative[strategy_item.start : strategy_item.end]
                latent_batch = latents[strategy_item.start : strategy_item.end]
                log.info(
                    f"start = {strategy_item.start}, end = {strategy_item.end} combine = {strategy_item.combine_cond_uncond}"
                )
                MemoryProfiler.record_memory(f"batch_{strategy_item.start}-{strategy_item.end}_before_inference_loop")

                # 计算当前batch的起始步骤偏移
                batch_step_offset = batch_idx * total_steps_per_batch
                bar_info = (batch_step_offset, global_total_steps)

                batch_sample_scheduler, _, batch_timesteps = get_sample_scheduler(**sample_scheduler_kwargs)
                log.info(f"batch timesteps: {batch_timesteps}, boundary:{boundary}, seq_len:{seq_len}")

                processed_latents = self._run_inference_loop(
                    positive_batch=pos_batch,
                    negative_batch=neg_batch,
                    latents_batch=latent_batch,
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

                latents[strategy_item.start : strategy_item.end] = processed_latents

        log.debug(f"latents shape: {latents.shape}")
        # [bs, 16, fi, hi, wi]
        return latents

    @time_range
    def forward_diffusion_models(
        self,
        model_pool: KsanaModelPool,
        positive: torch.Tensor,  # [bs, 512, 4096]
        negative: torch.Tensor,  # [bs, 512, 4096]
        sample_config: KsanaSampleConfig = None,
        runtime_config: KsanaRuntimeConfig = None,
        device: torch.device = None,
        offload_device: torch.device = None,
    ):
        log.info("start generate video")
        if model_pool is None:
            raise ValueError("model_pool must not be None")
        diffusion_models = model_pool.get_models(self.diffusion_model_keys)

        runtime_config = KsanaRuntimeConfig.copy_with_default(runtime_config, self.pipeline_config.default_config)
        high_cache_config, low_cache_config = self.process_input_cache(runtime_config.cache_method)

        latents = self.forward_diffusion_models_with_tensors(
            diffusion_models=diffusion_models,
            positive=positive,
            negative=negative,
            latents=None,
            sample_config=self.prepare_sample_default_args(sample_config),
            runtime_config=runtime_config,
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
            device=device,
            offload_device=offload_device,
        )
        del positive, negative

        # TODO: estimate diffusion memory usage to check whether neeed to offload diffusion model
        # if runtime_config.offload_model and offload_device is not None :
        # here always offload diffusion model to offload_device
        if offload_device:
            [diffusion_model.to(offload_device) for diffusion_model in diffusion_models]
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
