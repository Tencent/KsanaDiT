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

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from ..config.video_control_config import KsanaVideoControlConfig
from .experimental_sampling import (
    compute_cfg_zero_star_alpha,
    compute_raag_guidance,
    fourier_filter,
    tangential_projection,
    temporal_score_rescaling,
)
from .latent_format import get_wan21_latent_format
from .logger import log


@dataclass
class KsanaVaceVideoEncodeConfig:
    control_video: torch.Tensor | None = field(
        default=None, metadata={"help": "Control video frames [N, H, W, C] in range [0, 1]"}
    )
    control_masks: torch.Tensor | None = field(
        default=None, metadata={"help": "Mask tensor [N, H, W, 1] or [N, H, W], 1=editable, 0=preserve"}
    )
    reference_image: torch.Tensor | None = field(
        default=None, metadata={"help": "Reference image [1, H, W, C] in range [0, 1]"}
    )
    strength: float = field(default=1.0, metadata={"help": "VACE conditioning strength, typically 1.0"})

    vace_context: list[torch.Tensor] | None = field(default=None)
    vace_context_scale: float = field(default=1.0)
    trim_latent: int = field(default=0)
    adjusted_frame_num: int | None = field(
        default=None, metadata={"help": "Adjusted frame_num including reference image frames"}
    )
    vace_start_percent: float = field(default=0.0, metadata={"help": "Start percent of steps to apply VACE"})
    vace_end_percent: float = field(default=1.0, metadata={"help": "End percent of steps to apply VACE"})

    @property
    def has_control(self) -> bool:
        """Check if any user-provided control input exists."""
        return self.control_video is not None or self.reference_image is not None

    @property
    def num_frames(self) -> int | None:
        """Get the number of control video frames, if available."""
        if self.control_video is not None:
            return self.control_video.shape[0]
        return None

    @property
    def frame_size(self) -> tuple | None:
        """Get the (height, width) of control video frames, if available."""
        if self.control_video is not None:
            return (self.control_video.shape[1], self.control_video.shape[2])
        if self.reference_image is not None:
            return (self.reference_image.shape[1], self.reference_image.shape[2])
        return None


def resize_frames_torch(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Resize frames to target size using torch interpolation.

    Args:
        frames: Input tensor [N, H, W, C] in range [0, 1]
        width: Target width
        height: Target height

    Returns:
        Resized tensor [N, H, W, C]
    """
    # Convert to [N, C, H, W] for interpolation
    frames = frames.permute(0, 3, 1, 2)
    frames = torch.nn.functional.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)
    # Convert back to [N, H, W, C]
    return frames.permute(0, 2, 3, 1)


def pad_frames_torch(frames: torch.Tensor, target_length: int, pad_value: float = 0.5) -> torch.Tensor:
    """
    Pad frames to target length.

    Args:
        frames: Input tensor [N, H, W, C]
        target_length: Target number of frames
        pad_value: Value to pad with

    Returns:
        Padded tensor [target_length, H, W, C]
    """
    current_length = frames.shape[0]
    if current_length >= target_length:
        return frames[:target_length]
    padding = torch.full(
        (target_length - current_length, *frames.shape[1:]),
        pad_value,
        dtype=frames.dtype,
        device=frames.device,
    )
    return torch.cat([frames, padding], dim=0)


def build_vace_video_control_config(
    video_control_config: KsanaVaceVideoEncodeConfig | None,
    width: int,
    height: int,
    num_frames: int,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    vae_stride: int = 8,
) -> KsanaVaceVideoEncodeConfig | None:
    """
    Build VACE video control config by encoding control inputs.

    Args:
        video_control_config: Input config with control_video, reference_image, control_masks
        width: Target width (will be aligned to vae_stride)
        height: Target height (will be aligned to vae_stride)
        num_frames: Number of frames
        vae_encode_fn: Function to encode frames to latent, input [N,H,W,C] output [1,C,T,H,W]
        vae_stride: VAE stride for dimension alignment

    Returns:
        Config with vace_context populated, or None if no control
    """
    if video_control_config is None or not video_control_config.has_control:
        return None

    if video_control_config.vace_context is not None:
        return video_control_config

    # Align dimensions to VAE stride
    width = (width // vae_stride) * vae_stride
    height = (height // vae_stride) * vae_stride
    log.info(f"VACE target dimensions (aligned to {vae_stride}): {width}x{height}, frames={num_frames}")

    # Prefer GPU if available
    compute_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Prepare control video frames in [0, 1], [N, H, W, C]
    if video_control_config.control_video is not None:
        control_video = video_control_config.control_video
    else:
        control_video = torch.ones((num_frames, height, width, 3), dtype=torch.float32) * 0.5
    control_video = control_video.to(device=compute_device, dtype=torch.float32)

    reference_image = (
        video_control_config.reference_image.to(device=compute_device, dtype=torch.float32)
        if video_control_config.reference_image is not None
        else None
    )

    control_masks = (
        video_control_config.control_masks.to(device=compute_device, dtype=torch.float32)
        if video_control_config.control_masks is not None
        else None
    )

    vace_context, vace_strength, trim_latent = encode_vace_context(
        control_video=control_video,
        control_masks=control_masks,
        reference_image=reference_image,
        width=width,
        height=height,
        num_frames=num_frames,
        vae_encode_fn=vae_encode_fn,
        vace_strength=video_control_config.strength,
        normalize_latent=True,
    )

    log.info(
        "Built VACE control config: "
        f"vace_context_len={len(vace_context)}, "
        f"vace_context_shape={vace_context[0].shape if vace_context else 'none'}, "
        f"strength={vace_strength}, "
        f"trim_latent={trim_latent}"
    )

    return KsanaVaceVideoEncodeConfig(
        vace_context=vace_context,
        vace_context_scale=vace_strength,
        trim_latent=trim_latent,
    )


def encode_vace_context(
    control_video: torch.Tensor,
    control_masks: torch.Tensor | None,
    reference_image: torch.Tensor | None,
    width: int,
    height: int,
    num_frames: int,
    vae_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    vace_strength: float = 1.0,
    resize_fn: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
    normalize_latent: bool = True,
) -> tuple[list[torch.Tensor], float, int]:
    """
    Core VACE encoding logic - creates vace_context from inputs.

    This function is used by both:
    - KsanaWanVaceToVideoNode (ComfyUI)
    - Local script examples

    Args:
        control_video: Control video frames [N, H, W, C] in [0, 1]
        control_masks: Optional masks [N, H, W, 1] in [0, 1], 1=editable. None means all editable.
        reference_image: Optional reference image [1, H, W, C] in [0, 1]
        width: Target width
        height: Target height
        num_frames: Number of output frames
        vae_encode_fn: Function to encode frames with VAE: (frames [N,H,W,C]) -> latent [1,C,T,H,W]
        vace_strength: VACE conditioning strength
        resize_fn: Optional custom resize function. If None, uses torch interpolation.
        normalize_latent: Whether to apply process_in normalization to latent. Default True for
            local scripts (direct usage). Set to False for ComfyUI mode where normalization
            is done later in _extract_vace_from_conditioning.

    Returns:
        Tuple of:
        - vace_context: list of tensors [C, T, H, W] (usually single element)
        - vace_strength: float
        - trim_latent: int (frames to trim for reference image)
    """
    if resize_fn is None:
        resize_fn = resize_frames_torch

    latent_format = get_wan21_latent_format()
    device = control_video.device
    dtype = control_video.dtype
    length = num_frames
    latent_length_video = ((length - 1) // 4) + 1  # Latent frames for video only (excluding reference)

    # Resize and pad control_video
    control_video = resize_fn(control_video[:length], width, height)
    control_video = pad_frames_torch(control_video, length, pad_value=0.5)
    control_video = control_video.to(device=device, dtype=dtype)

    # Process masks
    if control_masks is None:
        mask = torch.ones((length, height, width, 1), device=device, dtype=dtype)
    else:
        mask = control_masks[:length]
        if mask.ndim == 3:
            mask = mask.unsqueeze(-1)
        mask = resize_fn(mask, width, height)
        mask = pad_frames_torch(mask, length, pad_value=1.0)
        mask = mask.to(device=device, dtype=dtype)

    # Create inactive and reactive based on mask
    # control_video is in [0, 1], center it to [-0.5, 0.5]
    control_video_centered = control_video - 0.5
    inactive = (control_video_centered * (1 - mask)) + 0.5  # Non-masked regions
    reactive = (control_video_centered * mask) + 0.5  # Masked (editable) regions

    # Encode with VAE
    inactive_latent = vae_encode_fn(inactive)
    reactive_latent = vae_encode_fn(reactive)

    # Concat inactive and reactive: [1, 32, T, H, W]
    control_video_latent = torch.cat((inactive_latent, reactive_latent), dim=1)

    # Process reference image if provided
    trim_latent = 0
    if reference_image is not None:
        reference_image = resize_fn(reference_image[:1], width, height)
        reference_image = reference_image.to(device=device, dtype=dtype)
        ref_latent = vae_encode_fn(reference_image)
        # Concat with zeros for reactive part
        ref_zeros = latent_format.process_out(torch.zeros_like(ref_latent))
        ref_latent = torch.cat([ref_latent, ref_zeros], dim=1)
        # Prepend reference to control_video_latent
        control_video_latent = torch.cat((ref_latent, control_video_latent), dim=2)
        trim_latent = ref_latent.shape[2]

    # Process mask for latent space
    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride

    # Reshape mask: [N, H, W, 1] -> [64, lat_T, lat_H, lat_W]
    mask_reshaped = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask_reshaped = mask_reshaped.permute(2, 4, 0, 1, 3)  # [8, 8, N, lat_H, lat_W]
    mask_reshaped = mask_reshaped.reshape(
        vae_stride * vae_stride, length, height_mask, width_mask
    )  # [64, N, lat_H, lat_W]
    # Interpolate mask to video latent length (excluding reference frames)
    mask_latent = torch.nn.functional.interpolate(
        mask_reshaped.unsqueeze(0),
        size=(latent_length_video, height_mask, width_mask),
        mode="nearest-exact",
    ).squeeze(0)

    if reference_image is not None:
        # Prepend zeros for reference frames (zeros = not editable)
        mask_pad = torch.zeros(
            (mask_latent.shape[0], trim_latent, height_mask, width_mask),
            device=mask_latent.device,
            dtype=mask_latent.dtype,
        )
        mask_latent = torch.cat((mask_pad, mask_latent), dim=1)

    # Apply latent format processing if requested (for local scripts that directly use vace_context)
    # For ComfyUI mode, set normalize_latent=False since _extract_vace_from_conditioning will do it
    vf = control_video_latent.clone()
    if normalize_latent:
        for i in range(0, vf.shape[1], 16):
            vf[:, i : i + 16] = latent_format.process_in(vf[:, i : i + 16])

    # Concat latent with mask: [1, 32, T, H, W] + [64, T, H, W] -> [1, 96, T, H, W]
    mask_latent = mask_latent.unsqueeze(0).to(vf.device, dtype=vf.dtype)
    vf = torch.cat([vf, mask_latent], dim=1)

    # Remove batch dim: [96, T, H, W]
    vace_context = [vf.squeeze(0)]

    return vace_context, vace_strength, trim_latent


def extract_vace_from_conditioning(
    positive,
    negative,
    latent_shape: tuple,
    model_config=None,
) -> tuple[list[torch.Tensor] | None, float, int]:
    """
    Extract VACE context from ComfyUI-style conditioning.

    This function extracts vace_frames, vace_mask, and vace_strength from
    conditioning dictionaries (typically set by ComfyUI nodes like KsanaWanVaceToVideoNode).

    Args:
        positive: Positive conditioning (ComfyUI format)
        negative: Negative conditioning (ComfyUI format, unused but kept for API consistency)
        latent_shape: Shape of the latent tensor [B, C, T, H, W]
        model_config: Optional model configuration

    Returns:
        Tuple of:
        - vace_context_list: List of tensors [C, T, H, W], or None if not found
        - vace_strength: float
        - trim_latent: int
    """
    if not positive or len(positive) == 0:
        return None, 1.0, 0

    cond_dict = {}
    if positive and isinstance(positive, (list, tuple)):
        for item in reversed(positive):
            if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict):
                d = item[1]
                if "vace_frames" in d or "vace_mask" in d or "vace_strength" in d:
                    cond_dict = d
                    break
        if not cond_dict and isinstance(positive[0], (list, tuple)) and len(positive[0]) > 1:
            cond_dict = positive[0][1] if isinstance(positive[0][1], dict) else {}

    vace_frames = cond_dict.get("vace_frames", None)
    if vace_frames is None:
        return None, 1.0, 0
    if isinstance(vace_frames, torch.Tensor):
        vace_frames = [vace_frames]

    vace_mask = cond_dict.get("vace_mask", None)
    if isinstance(vace_mask, torch.Tensor):
        vace_mask = [vace_mask]

    vace_strength = cond_dict.get("vace_strength", [1.0])
    if isinstance(vace_strength, (int, float)):
        vace_strength = [float(vace_strength)]

    noise_shape = list(latent_shape)
    cpu = torch.device("cpu")

    if vace_mask is None:
        # 64 mask channels by design for VACE conditioning
        noise_shape[1] = 64
        vace_mask = [torch.ones(noise_shape, device=cpu)] * len(vace_frames)

    vace_context_list = []
    latent_format = get_wan21_latent_format()
    for j in range(len(vace_frames)):
        vf = vace_frames[j].clone()
        for i in range(0, vf.shape[1], 16):
            vf[:, i : i + 16] = latent_format.process_in(vf[:, i : i + 16])
        vf = torch.cat([vf, vace_mask[j]], dim=1)
        vf = vf.squeeze(0)
        vace_context_list.append(vf)

    latent_time_dim = latent_shape[2] if len(latent_shape) >= 3 else 1
    vace_time_dim = (
        vace_frames[0].shape[2] if len(vace_frames) > 0 and len(vace_frames[0].shape) >= 3 else latent_time_dim
    )
    # trim_latent should be the difference between vace_context time dimension and noise latent time dimension
    # This represents the number of reference image frames that were prepended to vace_context
    trim_latent = max(0, vace_time_dim - latent_time_dim)

    log.info("=" * 60)
    log.info("[extract_vace_from_conditioning] Results:")
    log.info(f"  positive_len: {len(positive) if hasattr(positive, '__len__') else 'n/a'}")
    log.info(f"  cond_keys: {list(cond_dict.keys())}")
    log.info(f"  latent_shape (noise): {latent_shape}")
    log.info(f"  num_vace_frames: {len(vace_frames) if hasattr(vace_frames, '__len__') else 'n/a'}")
    log.info(f"  vace_frames shapes: {[vf.shape for vf in vace_frames]}")
    log.info(f"  vace_mask shapes: {[vm.shape for vm in vace_mask] if vace_mask else 'None'}")
    log.info(f"  vace_strength: {vace_strength}")
    log.info(f"  vace_context_list shapes: {[vc.shape for vc in vace_context_list]}")
    log.info(f"  latent_time_dim: {latent_time_dim}, vace_time_dim: {vace_time_dim}")
    log.info(f"  trim_latent: {trim_latent}")
    log.info("=" * 60)
    return vace_context_list, vace_strength[0] if len(vace_strength) == 1 else vace_strength, trim_latent


def build_vace_kwargs(
    control_video_config: KsanaVaceVideoEncodeConfig | None,
    noise_shape: tuple,
    device,
    sample_config,
    video_control: KsanaVideoControlConfig | None = None,
) -> dict:
    result = {}
    if control_video_config is not None:
        vace_context = control_video_config.vace_context
        vace_context_scale = control_video_config.vace_context_scale
        trim_latent = control_video_config.trim_latent
        vace_start_percent = control_video_config.vace_start_percent
        vace_end_percent = control_video_config.vace_end_percent

        if vace_context is not None:
            vace_context = [vc.to(device) if vc.device != device else vc for vc in vace_context]
            result.update(
                {
                    "vace_context": vace_context,
                    "vace_context_scale": vace_context_scale,
                    "trim_latent": trim_latent,
                    "vace_start_percent": vace_start_percent,
                    "vace_end_percent": vace_end_percent,
                }
            )

    base_video_control = sample_config.video_control
    effective_slg_args = (
        video_control.slg
        if video_control and video_control.slg is not None
        else (base_video_control.slg if base_video_control else None)
    )
    effective_feta_args = (
        video_control.feta
        if video_control and video_control.feta is not None
        else (base_video_control.feta if base_video_control else None)
    )
    effective_experimental_args = (
        video_control.experimental
        if video_control and video_control.experimental is not None
        else (base_video_control.experimental if base_video_control else None)
    )

    if effective_slg_args is not None:
        log.info(f"Using SLG config: {effective_slg_args}")
    if effective_feta_args is not None:
        log.info(f"Using FETA config: {effective_feta_args}")
    if effective_experimental_args is not None:
        log.info(f"Using experimental config: {effective_experimental_args}")

    result.update(
        {
            "slg_args": effective_slg_args,
            "feta_args": effective_feta_args,
            "experimental_args": effective_experimental_args,
        }
    )

    return result


def parse_video_control_kwargs(
    video_control_kwargs: dict | None,
    diffusion_model: list,
    sample_scheduler=None,
    slg_config_cls=None,
    feta_config_cls=None,
    experimental_config_cls=None,
) -> dict:
    """Parse video control kwargs into structured config.

    Args:
        video_control_kwargs: Dict with slg_args, feta_args, experimental_args, vace_* fields
        diffusion_model: List of diffusion models to apply video_attention_split_steps
        sample_scheduler: Scheduler for bidirectional sampling
        slg_config_cls: KsanaSLGConfig class
        feta_config_cls: KsanaFETAConfig class
        experimental_config_cls: KsanaExperimentalConfig class
    """
    video_control_kwargs = video_control_kwargs or {}
    slg_args = video_control_kwargs.get("slg_args")
    feta_args = video_control_kwargs.get("feta_args")
    experimental_args = video_control_kwargs.get("experimental_args")

    # Parse SLG config
    slg_blocks = []
    slg_start_percent = 0.0
    slg_end_percent = 1.0
    if slg_args is not None:
        if slg_config_cls and isinstance(slg_args, slg_config_cls):
            slg_blocks = slg_args.blocks or []
            slg_start_percent = slg_args.start_percent
            slg_end_percent = slg_args.end_percent
        elif isinstance(slg_args, dict):
            slg_blocks = slg_args.get("blocks", [])
            slg_start_percent = slg_args.get("start_percent", 0.0)
            slg_end_percent = slg_args.get("end_percent", 1.0)
        log.info(f"[VideoControl] SLG: blocks={slg_blocks}, range=[{slg_start_percent}, {slg_end_percent}]")

    # Parse FETA config
    feta_enabled_range = (0.0, 1.0)
    feta_weight = 2.0
    if feta_args is not None:
        if feta_config_cls and isinstance(feta_args, feta_config_cls):
            feta_enabled_range = (feta_args.start_percent, feta_args.end_percent)
            feta_weight = feta_args.weight
        elif isinstance(feta_args, dict):
            feta_enabled_range = (feta_args.get("start_percent", 0.0), feta_args.get("end_percent", 1.0))
            feta_weight = feta_args.get("weight", 2.0)
        log.info(f"[VideoControl] FETA: weight={feta_weight}, range={feta_enabled_range}")

    # Parse experimental config
    exp_config = None
    if experimental_args is not None:
        if experimental_config_cls and isinstance(experimental_args, experimental_config_cls):
            exp_config = experimental_args
        elif isinstance(experimental_args, dict) and experimental_config_cls:
            exp_config = experimental_config_cls(
                cfg_zero_star=experimental_args.get("cfg_zero_star", False),
                use_zero_init=experimental_args.get("use_zero_init", False),
                zero_star_steps=experimental_args.get("zero_star_steps", 0),
                use_fresca=experimental_args.get("use_fresca", False),
                fresca_scale_low=experimental_args.get("fresca_scale_low", 1.0),
                fresca_scale_high=experimental_args.get("fresca_scale_high", 1.25),
                fresca_freq_cutoff=experimental_args.get("fresca_freq_cutoff", 20),
                use_tcfg=experimental_args.get("use_tcfg", False),
                raag_alpha=experimental_args.get("raag_alpha", 0.0),
                bidirectional_sampling=experimental_args.get("bidirectional_sampling", False),
                temporal_score_rescaling=experimental_args.get("temporal_score_rescaling", False),
                tsr_k=experimental_args.get("tsr_k", 0.95),
                tsr_sigma=experimental_args.get("tsr_sigma", 1.0),
                video_attention_split_steps=experimental_args.get("video_attention_split_steps", ""),
            )
        log.info(f"[VideoControl] Experimental: {exp_config}")

    # Parse bidirectional sampling
    bidirectional_sampling = False
    sample_scheduler_flipped = None
    if exp_config is not None and exp_config.bidirectional_sampling:
        bidirectional_sampling = True
        sample_scheduler_flipped = copy.deepcopy(sample_scheduler)
        log.info("[VideoControl] Bidirectional sampling enabled")

    # Parse video_attention_split_steps and apply to models
    if exp_config is not None and exp_config.video_attention_split_steps:
        steps_str = exp_config.video_attention_split_steps
        if isinstance(steps_str, str) and steps_str.strip():
            video_attention_split_steps = [int(x.strip()) for x in steps_str.split(",") if x.strip()]
            log.info(f"[VideoControl] video_attention_split_steps: {video_attention_split_steps}")
            for model in diffusion_model:
                if hasattr(model, "model") and hasattr(model.model, "video_attention_split_steps"):
                    model.model.video_attention_split_steps = video_attention_split_steps

    return {
        "slg_blocks": slg_blocks,
        "slg_start_percent": slg_start_percent,
        "slg_end_percent": slg_end_percent,
        "feta_enabled": feta_args is not None,
        "feta_enabled_range": feta_enabled_range,
        "feta_weight": feta_weight,
        "exp_config": exp_config,
        "bidirectional_sampling": bidirectional_sampling,
        "sample_scheduler_flipped": sample_scheduler_flipped,
        "vace_context": video_control_kwargs.get("vace_context"),
        "vace_context_scale": video_control_kwargs.get("vace_context_scale", 1.0),
        "vace_start_percent": video_control_kwargs.get("vace_start_percent", 0.0),
        "vace_end_percent": video_control_kwargs.get("vace_end_percent", 1.0),
    }


def get_step_video_control(
    vc: dict,
    current_step_percent: float,
    iter_id: int,
    total_steps: int,
    slg_config_cls=None,
    feta_config_cls=None,
) -> dict:
    """Get video control configs for current step.

    Returns dict with: slg_config, feta_config, vace_context, vace_context_scale, current_step_percent
    """
    slg_blocks = vc["slg_blocks"]
    slg_start_percent = vc["slg_start_percent"]
    slg_end_percent = vc["slg_end_percent"]
    feta_enabled_range = vc["feta_enabled_range"]
    feta_weight = vc["feta_weight"]
    vace_context = vc["vace_context"]
    vace_context_scale = vc["vace_context_scale"]
    vace_start_percent = vc["vace_start_percent"]
    vace_end_percent = vc["vace_end_percent"]

    should_log = iter_id == 0 or iter_id == total_steps - 1 or iter_id % 5 == 0

    # SLG config
    slg_config = None
    slg_active = slg_blocks and slg_start_percent <= current_step_percent <= slg_end_percent
    if slg_active and slg_config_cls:
        slg_config = slg_config_cls(blocks=slg_blocks, start_percent=slg_start_percent, end_percent=slg_end_percent)
    if should_log and slg_blocks:
        log.info(f"[Step {iter_id}/{total_steps}] SLG: active={slg_active}, percent={current_step_percent:.3f}")

    # FETA config
    feta_config = None
    feta_enabled = vc["feta_enabled"]
    feta_active = feta_enabled and feta_enabled_range[0] <= current_step_percent <= feta_enabled_range[1]
    if feta_active and feta_config_cls:
        feta_config = feta_config_cls(
            start_percent=feta_enabled_range[0], end_percent=feta_enabled_range[1], weight=feta_weight
        )
    if should_log and feta_enabled:
        log.info(f"[Step {iter_id}/{total_steps}] FETA: active={feta_active}, weight={feta_weight}")

    # VACE config
    vace_active = vace_context is not None and vace_start_percent <= current_step_percent <= vace_end_percent
    current_vace_context = vace_context if vace_active else None
    current_vace_scale = vace_context_scale if vace_active else 1.0
    if should_log and vace_context is not None:
        log.info(f"[Step {iter_id}/{total_steps}] VACE: active={vace_active}, percent={current_step_percent:.3f}")

    return {
        "slg_config": slg_config,
        "feta_config": feta_config,
        "vace_context": current_vace_context,
        "vace_context_scale": current_vace_scale,
        "current_step_percent": current_step_percent,
    }


def apply_vace_trim(latents: torch.Tensor, trim_latent: int) -> torch.Tensor:
    """Trim reference image frames from latent output."""
    if trim_latent > 0 and latents.dim() >= 3:
        log.info(f"Trimming {trim_latent} reference frames from latent. Original shape: {latents.shape}")
        latents = latents[:, :, trim_latent:, :, :]
    return latents


def apply_temporal_score_rescaling(
    noise_pred: torch.Tensor,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    exp_config,
) -> torch.Tensor:
    """Apply Temporal Score Rescaling (TSR) if enabled."""
    if exp_config is not None and exp_config.temporal_score_rescaling:
        return temporal_score_rescaling(
            noise_pred=noise_pred,
            latent=latent,
            timestep=timestep,
            tsr_k=exp_config.tsr_k,
            tsr_sigma=exp_config.tsr_sigma,
        )
    return noise_pred


def apply_experimental_cfg(
    cfg_scale: float,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    exp_config,
    step_index: int,
) -> torch.Tensor:
    """Apply experimental CFG techniques (CFG-Zero-Star, TCFG, RAAG, FreSca)."""
    should_log = step_index % 5 == 0

    # Log config on first step
    if step_index == 0:
        log.info("=" * 60)
        log.info("[apply_cfg] EXPERIMENTAL CONFIG ACTIVE:")
        log.info(f"  cfg_zero_star={exp_config.cfg_zero_star}, use_zero_init={exp_config.use_zero_init}")
        log.info(f"  use_fresca={exp_config.use_fresca}, use_tcfg={exp_config.use_tcfg}")
        log.info(f"  raag_alpha={exp_config.raag_alpha}")
        log.info("=" * 60)

    # CFG-Zero-Star: zero init for early steps
    if exp_config.cfg_zero_star:
        if exp_config.use_zero_init and step_index < exp_config.zero_star_steps:
            if step_index == 0:
                log.info(f"[Step {step_index}] CFG-Zero-Star: Returning zero noise")
            return torch.zeros_like(cond)
        alpha = compute_cfg_zero_star_alpha(cond, uncond)
        uncond = alpha * uncond
        if should_log:
            log.info(f"[Step {step_index}] CFG-Zero-Star: alpha={alpha.item():.4f}")

    # TCFG: tangential projection
    if exp_config.use_tcfg:
        uncond = tangential_projection(cond, uncond)
        if should_log:
            log.info(f"[Step {step_index}] TCFG: Applied")

    # RAAG: ratio-aware adaptive guidance
    effective_cfg_scale = float(cfg_scale)
    if exp_config.raag_alpha > 0:
        effective_cfg_scale = compute_raag_guidance(cond, uncond, cfg_scale, exp_config.raag_alpha)
        if should_log:
            log.info(f"[Step {step_index}] RAAG: cfg_scale={effective_cfg_scale:.4f}")

    cfg_diff = cond - uncond

    # FreSca: frequency-domain scaling
    if exp_config.use_fresca:
        cfg_diff = fourier_filter(
            cfg_diff,
            scale_low=exp_config.fresca_scale_low,
            scale_high=exp_config.fresca_scale_high,
            freq_cutoff=exp_config.fresca_freq_cutoff,
        )
        if should_log:
            log.info(f"[Step {step_index}] FreSca: Applied")

    noise_pred = uncond + effective_cfg_scale * cfg_diff

    # NaN check
    if torch.isnan(noise_pred).any():
        log.warning(f"[Step {step_index}] NaN detected, falling back to cond")
        noise_pred = torch.nan_to_num(cond, nan=0.0)

    return noise_pred


def apply_bidirectional_sampling(
    noise_latent: torch.Tensor,
    noise_latent_forward: torch.Tensor,
    running_model,
    running_cfg_scale: float,
    timestep: torch.Tensor,
    t: torch.Tensor,
    iter_id: int,
    total_steps: int,
    current_step_percent: float,
    combine_cond_uncond: bool,
    positive,
    negative,
    img_latent,
    step_vc: dict,
    exp_config,
    sample_scheduler_flipped,
    sample_config,
    seed_g: torch.Generator,
    prepare_model_forward_kargs_fn,
    use_cfg_fn,
    apply_cfg_fn,
    solver_type_euler,
) -> torch.Tensor:
    """Apply bidirectional sampling: flip, infer, average with forward result.

    Args:
        prepare_model_forward_kargs_fn: Callback to prepare forward kwargs
        use_cfg_fn: Callback to check if CFG is enabled
        apply_cfg_fn: Callback to apply CFG
        solver_type_euler: The EULER solver type enum value for comparison
    """
    # Flip latent along time dimension (dim=2 for [B, C, T, H, W])
    noise_latent_flipped = torch.flip(noise_latent, dims=[2])

    # Prepare forward kwargs for flipped latent
    forward_kargs_flipped = prepare_model_forward_kargs_fn(
        running_cfg_scale,
        noise_latent=noise_latent_flipped,
        timestep=timestep,
        combine_cond_uncond=combine_cond_uncond,
        step_iter=iter_id,
        cache=None,  # Don't use cache for flipped pass
        positive=positive,
        negative=negative,
        img_latent=img_latent,
        **step_vc,
    )

    # Run model on flipped latent
    if use_cfg_fn(running_cfg_scale):
        if combine_cond_uncond:
            noise_pred_batch_flipped = running_model.forward(**forward_kargs_flipped)
            noise_pred_cond_flipped, noise_pred_uncond_flipped = noise_pred_batch_flipped.chunk(2, dim=0)
        else:
            arg_cond_flipped, arg_uncond_flipped = forward_kargs_flipped
            noise_pred_cond_flipped = running_model.forward(**arg_cond_flipped)
            noise_pred_uncond_flipped = running_model.forward(**arg_uncond_flipped)
        noise_pred_flipped = apply_cfg_fn(
            running_cfg_scale,
            noise_pred_cond_flipped,
            noise_pred_uncond_flipped,
            experimental_config=exp_config,
            step_index=iter_id,
            total_steps=total_steps,
        )
    else:
        noise_pred_flipped = running_model.forward(**forward_kargs_flipped)

    # Apply TSR to flipped prediction
    noise_pred_flipped = apply_temporal_score_rescaling(noise_pred_flipped, noise_latent_flipped, t, exp_config)

    # Step with flipped prediction using the copied scheduler
    step_out_flipped = sample_scheduler_flipped.step(
        noise_pred_flipped, t, noise_latent_flipped, return_dict=False, generator=seed_g
    )
    noise_latent_backward = step_out_flipped if sample_config.solver == solver_type_euler else step_out_flipped[0]
    noise_latent_backward = noise_latent_backward.reshape(noise_latent.shape)

    # Flip backward result back and average with forward result
    noise_latent_backward = torch.flip(noise_latent_backward, dims=[2])
    result = noise_latent_forward * 0.5 + noise_latent_backward * 0.5

    if iter_id % 5 == 0:
        log.info(f"[Step {iter_id}] Bidirectional Sampling: Applied (forward + backward avg)")

    return result


def prepare_video_control_config(video_control_config=None, vace_embeds=None):
    video_control = None
    log.info(f"video_control_config received: {video_control_config}")
    if video_control_config is not None:
        slg_args = video_control_config.get("slg_args")
        feta_args = video_control_config.get("feta_args")
        experimental_args = video_control_config.get("experimental_args")
        log.info(
            f"Extracted from video_control_config: slg_args={slg_args}, "
            f"feta_args={feta_args}, experimental_args={experimental_args}"
        )
        if slg_args is not None or feta_args is not None or experimental_args is not None:
            video_control = KsanaVideoControlConfig(
                slg=slg_args,
                feta=feta_args,
                experimental=experimental_args,
            )

    control_video_config = None
    if vace_embeds is not None:
        log.info("Using vace_embeds directly (decoupled VACE input)")
        vace_frames = vace_embeds.get("vace_frames")
        vace_mask = vace_embeds.get("vace_mask")
        vace_scale = vace_embeds.get("vace_scale", 1.0)
        trim_latent = vace_embeds.get("trim_latent", 0)
        vace_start_percent = vace_embeds.get("vace_start_percent", 0.0)
        vace_end_percent = vace_embeds.get("vace_end_percent", 1.0)
        additional_vace_inputs = vace_embeds.get("additional_vace_inputs", [])

        if vace_frames:
            vf_shapes = [v.shape for v in vace_frames] if isinstance(vace_frames, list) else vace_frames.shape
            log.info(f"  vace_frames shapes: {vf_shapes}")
        if vace_mask:
            vm_shapes = [v.shape for v in vace_mask] if isinstance(vace_mask, list) else vace_mask.shape
            log.info(f"  vace_mask shapes: {vm_shapes}")
        log.info(f"  vace_scale: {vace_scale}, trim_latent: {trim_latent}")
        log.info(f"  vace_start_percent: {vace_start_percent}, vace_end_percent: {vace_end_percent}")
        if additional_vace_inputs:
            log.info(f"  additional_vace_inputs count: {len(additional_vace_inputs)}")

        latent_format = get_wan21_latent_format()
        vace_context_list = []
        for j in range(len(vace_frames)):
            vf = vace_frames[j].clone()
            for i in range(0, vf.shape[1], 16):
                vf[:, i : i + 16] = latent_format.process_in(vf[:, i : i + 16])
            vf = vf.squeeze(0) if vf.dim() > 4 else vf
            if vace_mask and j < len(vace_mask):
                mask_j = vace_mask[j]
                vf = torch.cat([vf, mask_j.squeeze(0) if mask_j.dim() > 4 else mask_j], dim=0)
            vace_context_list.append(vf.squeeze(0) if vf.dim() > 3 else vf)

        control_video_config = KsanaVaceVideoEncodeConfig(
            vace_context=vace_context_list,
            vace_context_scale=vace_scale,
            trim_latent=trim_latent,
            vace_start_percent=vace_start_percent,
            vace_end_percent=vace_end_percent,
        )

    return video_control, control_video_config
