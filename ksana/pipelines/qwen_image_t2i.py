"""
Reference (Diffusers):
  - diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
"""

import os
import random
import sys
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from ..config import KsanaModelConfig, KsanaPipelineConfig, KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverBackend
from ..models import KsanaQwenImageModel
from ..models.model_pool import KsanaModelPool
from ..models.qwen import KsanaQwen2VLTextEncoder
from ..models.vae import KsanaVAE
from ..sample_solvers import calculate_shift, get_sample_scheduler
from ..utils import log, time_range
from ..utils.load import load_state_dict_from_path
from .base_x2x import KsanaDefaultArgs, KsanaX2XPipeline


@dataclass(frozen=True)
class QwenImageDefaultArgs(KsanaDefaultArgs):
    steps: int = field(default=50)
    cfg_scale: float = field(default=4.0)
    sample_solver: KsanaSolverBackend = field(default=KsanaSolverBackend.FLOWMATCH_EULER)


class KsanaQwenImageT2IPipeline(KsanaX2XPipeline):
    def __init__(self, pipeline_config: KsanaPipelineConfig):
        super().__init__(pipeline_config)
        self.default_args = QwenImageDefaultArgs()
        default_cfg = getattr(self.pipeline_config, "default_config", None)
        if default_cfg is not None:
            if hasattr(default_cfg, "z_dim"):
                self.vae_z_dim = default_cfg.z_dim
            if hasattr(default_cfg, "vae_stride"):
                self.vae_stride = default_cfg.vae_stride
            self.vae_scale_factor = getattr(default_cfg, "vae_scale_factor", 8)
        else:
            self.vae_scale_factor = 8

    def load_text_encoder(self, checkpoint_dir, shard_fn=None):
        text_encoder = KsanaQwen2VLTextEncoder(
            checkpoint_dir=checkpoint_dir,
            dtype=torch.bfloat16,
        )
        return text_encoder

    def load_vae(self, checkpoint_dir, device):
        vae = KsanaVAE(model_path=checkpoint_dir, device=device, dtype=torch.bfloat16)
        self.vae_z_dim = vae.z_dim
        self.vae_stride = vae.vae_stride
        return vae

    @time_range
    def load_diffusion_model(
        self,
        model_path,
        *,
        lora=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        input_model_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
        **kwargs,
    ):
        model = KsanaQwenImageModel(model_config, self.pipeline_config, dist_config)
        load_device = str(device) if device is not None else "cuda"
        default_cfg = self.pipeline_config.default_config
        if os.path.isfile(model_path):
            state_dict = load_state_dict_from_path(model_path, device=load_device)
        else:
            transformer_dir = os.path.join(model_path, default_cfg.transformer_subdir)
            state_dict = load_state_dict_from_path(transformer_dir, device=load_device)

        model.load(
            model_state_dict=state_dict,
            input_model_config=default_cfg,
            load_device=device,
            offload_device=offload_device,
        )
        return [model]

    def process_input_cache(self, cache_method):
        return None, None

    def forward_text_encoder(
        self,
        model_pool: KsanaModelPool,
        prompts_positive,
        prompts_negative=None,
        device=None,
        offload_device=None,
        offload_model=False,
    ):
        bs = len(prompts_positive)
        text_encoder = model_pool.get_model(self.text_encoder_key)
        default_neg_prompt = self.pipeline_config.default_config.sample_neg_prompt
        prompts_negative = prompts_negative if prompts_negative is not None else [default_neg_prompt] * bs

        if len(prompts_positive) != len(prompts_negative):
            raise RuntimeError(
                f"The number of negative prompts ({len(prompts_negative)}) "
                f"must match the number of positive prompts ({bs})."
            )

        if text_encoder.device != device:
            text_encoder.to(device)

        positive_embeds, positive_mask = text_encoder.forward(prompts_positive, device=device)
        negative_embeds, negative_mask = text_encoder.forward(prompts_negative, device=device)

        if offload_model:
            target_offload = offload_device or torch.device("cpu")
            text_encoder.to(target_offload)

        return (positive_embeds, positive_mask), (negative_embeds, negative_mask)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
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

    @time_range
    def forward_diffusion_models_with_tensors(
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
            num_train_timesteps=default_config.num_train_timesteps,
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

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        return latents

    def forward_vae(
        self,
        model_pool: KsanaModelPool,
        latents,
        local_rank,
        device=None,
        offload_device=None,
        offload_model=False,
        with_end_image=False,
    ):
        if local_rank != 0:
            return None

        vae_model = model_pool.get_model(self.vae_key)
        if vae_model.device != device:
            vae_model.to(device)

        latents = latents.to(vae_model.dtype)
        image = vae_model.decode(latents)

        # [B, C, T, H, W] -> [B, C, H, W]
        if image.dim() == 5:
            image = image[:, :, 0]
        image = (image + 1) / 2
        image = image.clamp(0, 1)

        if offload_model and offload_device is not None:
            vae_model.to(offload_device)

        log.info(f"Generated image shape: {image.shape}")
        return image
