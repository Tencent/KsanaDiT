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

import torch

from ksana import get_engine
from ksana.config import KsanaExperimentalConfig, KsanaFETAConfig, KsanaSLGConfig
from ksana.nodes import (
    KSANA_EXPERIMENTAL_ARGS,
    KSANA_FETA_ARGS,
    KSANA_SLG_ARGS,
    KSANA_VACE_EMBEDS,
    KSANA_VAE_ENCODE_OUTPUT,
    KSANA_VAE_MODEL,
    KSANA_VIDEO_CONTROL_CONFIG,
    WANVIDEO_EXPERIMENTAL_ARGS,
    WANVIDEO_FETA_ARGS,
    WANVIDEO_SLG_ARGS,
)
from ksana.nodes.output_types import KsanaNodeVAEEncodeOutput
from ksana.utils import common_upscale, get_intermediate_device
from ksana.utils.latent_format import get_wan21_latent_format
from ksana.utils.logger import log


class KsanaWanVaceToVideoNode:
    @classmethod
    def input_types(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Output video width.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 16,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Output video height.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Strength of the VACE conditioning.",
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Number of frames to generate.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "Batch size for generation.",
                    },
                ),
                "vace_start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start percent of the steps to apply VACE.",
                    },
                ),
                "vace_end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End percent of the steps to apply VACE.",
                    },
                ),
            },
            "optional": {
                "vae": (KSANA_VAE_MODEL, {"tooltip": "KsanaDiT VAE model for encoding."}),
                "control_video": (
                    "IMAGE",
                    {"tooltip": "Control video frames for VACE conditioning."},
                ),
                "control_masks": (
                    "MASK",
                    {"tooltip": "Masks for control video frames."},
                ),
                "reference_image": (
                    "IMAGE",
                    {"tooltip": "Reference image for style/content guidance."},
                ),
                "prev_vace_embeds": (
                    KSANA_VACE_EMBEDS,
                    {"tooltip": "Previous VACE embeddings for chaining multiple VACE inputs."},
                ),
            },
        }

    INPUT_TYPES = input_types  # Alias for ComfyUI compatibility

    RETURN_TYPES = (KSANA_VACE_EMBEDS, KSANA_VAE_ENCODE_OUTPUT)
    RETURN_NAMES = ("vace_embeds", "image_embeds")
    OUTPUT_TOOLTIPS = (
        "VACE embeddings containing vace_context, vace_scale, and metadata.",
        "Latent tensor for the video.",
    )
    FUNCTION = "encode"
    CATEGORY = "ksana/vace"
    DESCRIPTION = "Encodes control video and reference image for Wan VACE video generation using KsanaDiT VAE."

    def _vae_encode(self, vae_key, image):
        """Encode images using KsanaDiT VAE."""
        ksana_engine = get_engine()
        result = ksana_engine.forward_vae_encode_image(model_key=vae_key, image=image)
        return result

    def encode(
        self,
        width,
        height,
        strength,
        num_frames,
        batch_size,
        vace_start_percent,
        vace_end_percent,
        vae=None,
        control_video=None,
        control_masks=None,
        reference_image=None,
        prev_vace_embeds=None,
    ):
        log.info("[KsanaWanVaceToVideoNode] Starting encode...")
        log.info(
            f"  Input params: width={width}, height={height}, "
            f"num_frames={num_frames}, batch_size={batch_size}, strength={strength}"
        )
        log.info(f"  vace_start_percent: {vace_start_percent}, vace_end_percent: {vace_end_percent}")
        log.info(f"  control_video: {control_video.shape if control_video is not None else None}")
        log.info(f"  control_masks: {control_masks.shape if control_masks is not None else None}")
        log.info(f"  reference_image: {reference_image.shape if reference_image is not None else None}")
        log.info(f"  prev_vace_embeds: {prev_vace_embeds is not None}")

        if vae is None:
            raise ValueError("'vae' (KSANA_VAE_MODEL) must be connected.")

        fallback_device = get_intermediate_device()
        if isinstance(control_video, torch.Tensor):
            work_device = control_video.device
            work_dtype = control_video.dtype
        elif isinstance(reference_image, torch.Tensor):
            work_device = reference_image.device
            work_dtype = reference_image.dtype
        else:
            work_device = fallback_device
            work_dtype = torch.float32

        length = num_frames
        latent_length = ((length - 1) // 4) + 1

        if control_video is not None:
            control_video = common_upscale(
                control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3), device=work_device, dtype=work_dtype) * 0.5

        if isinstance(control_video, torch.Tensor):
            control_video = control_video.to(device=work_device, dtype=work_dtype)

        # Process reference_image
        if reference_image is not None:
            reference_image = common_upscale(
                reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            reference_image = self._vae_encode(vae, reference_image)
            reference_image = torch.cat(
                [reference_image, get_wan21_latent_format().process_out(torch.zeros_like(reference_image))],
                dim=1,
            )

        # Process control_masks
        if control_masks is None:
            mask = torch.ones((length, height, width, 1), device=work_device, dtype=work_dtype)
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)
            if isinstance(mask, torch.Tensor):
                mask = mask.to(device=work_device, dtype=work_dtype)

        # Generate inactive and reactive latent representations
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = self._vae_encode(vae, inactive)
        reactive = self._vae_encode(vae, reactive)
        control_video_latent = torch.cat((inactive, reactive), dim=1)

        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        # Process mask for latent space
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
        ).squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, : reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)
        mask = mask.to(device=control_video_latent.device, dtype=control_video_latent.dtype)

        # Build vace_embeds output (similar to WanVideoWrapper format)
        vace_embeds = {
            "vace_frames": [control_video_latent],
            "vace_mask": [mask],
            "vace_scale": strength,
            "has_ref": reference_image is not None,
            "num_frames": num_frames,
            "trim_latent": trim_latent,
            "target_shape": (16, latent_length, height // 8, width // 8),
            "vace_start_percent": vace_start_percent,
            "vace_end_percent": vace_end_percent,
            "additional_vace_inputs": [],
        }

        # Handle prev_vace_embeds for chaining multiple VACE inputs
        if prev_vace_embeds is not None:
            if "additional_vace_inputs" in prev_vace_embeds and prev_vace_embeds["additional_vace_inputs"]:
                vace_embeds["additional_vace_inputs"] = prev_vace_embeds["additional_vace_inputs"].copy()
            vace_embeds["additional_vace_inputs"].append(prev_vace_embeds)

        # Create output latent
        latent = torch.zeros(
            [batch_size, 16, latent_length, height // 8, width // 8],
            device=get_intermediate_device(),
        )
        out_latent = KsanaNodeVAEEncodeOutput(
            samples=latent,
            with_end_image=False,
            batch_size_per_prompts=batch_size,
        )

        log.info("[KsanaWanVaceToVideoNode] Encode complete:")
        log.info(f"  control_video_latent shape: {control_video_latent.shape}")
        log.info(f"  mask shape: {mask.shape}")
        log.info(f"  output latent shape: {latent.shape}")
        log.info(f"  latent_length: {latent_length}, trim_latent: {trim_latent}")
        log.info(f"  vace_start_percent: {vace_start_percent}, vace_end_percent: {vace_end_percent}")
        log.info(f"  vace_embeds keys: {list(vace_embeds.keys())}")

        return (vace_embeds, out_latent)


class KsanaSLGNode:
    """
    Skip Layer Guidance (SLG) Node.

    SLG is a sampling optimization that skips unconditional (negative prompt) inference
    on specified transformer blocks during CFG computation. This can significantly
    speed up generation while maintaining quality, as not all blocks contribute
    equally to the CFG signal.

    Typical usage: For Wan 14B models, blocks like 9, 10, or 11 often work well.
    """

    @classmethod
    def input_types(s):
        return {
            "required": {
                "blocks": (
                    "STRING",
                    {
                        "default": "10",
                        "tooltip": "Comma-separated transformer block indices (0-indexed) to skip uncond. "
                        "Example: '9,10,11' for Wan 14B model.",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start percentage of steps to enable SLG. 0.1 = after 10% of steps.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End percentage of steps to enable SLG. 1.0 = until final step.",
                    },
                ),
            },
        }

    INPUT_TYPES = input_types  # Alias for ComfyUI compatibility

    RETURN_TYPES = (KSANA_SLG_ARGS,)
    RETURN_NAMES = ("slg_args",)
    FUNCTION = "create_config"
    CATEGORY = "ksana/vace"
    DESCRIPTION = (
        "Skip Layer Guidance: Skips uncond inference on specified blocks to speed up sampling. "
        "Reference: Similar to techniques used in various CFG optimization papers."
    )

    def create_config(self, blocks: str, start_percent: float, end_percent: float):
        # Parse blocks string to list of integers
        block_list = [int(x.strip()) for x in blocks.split(",") if x.strip()]
        config = KsanaSLGConfig(
            blocks=block_list,
            start_percent=start_percent,
            end_percent=end_percent,
        )
        log.info("=" * 60)
        log.info("[KsanaSLGNode] CREATE_CONFIG CALLED - Skip Layer Guidance")
        log.info(f"  blocks: {block_list}")
        log.info(f"  start_percent: {start_percent}, end_percent: {end_percent}")
        log.info(f"  Created config: {config}")
        log.info("=" * 60)
        return (config,)


class KsanaEnhanceAVideoNode:
    """
    Enhance-A-Video (FETA) Node.

    FETA (Frame-Enhanced Temporal Attention) improves video temporal consistency
    by computing cross-frame attention scores and using them to modulate
    self-attention outputs. This reduces flickering and improves coherence.

    Reference: https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video
    """

    @classmethod
    def input_types(s):
        return {
            "required": {
                "weight": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Enhancement weight. Higher = stronger temporal smoothing. Typical range: 1.0-5.0",
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Start percentage of steps to enable FETA.",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "End percentage of steps to enable FETA.",
                    },
                ),
            },
        }

    INPUT_TYPES = input_types  # Alias for ComfyUI compatibility

    RETURN_TYPES = (KSANA_FETA_ARGS,)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "create_config"
    CATEGORY = "ksana/vace"
    DESCRIPTION = (
        "Enhance-A-Video: Improves temporal consistency by modulating attention with cross-frame scores. "
        "Reference: https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"
    )

    def create_config(self, weight: float, start_percent: float, end_percent: float):
        config = KsanaFETAConfig(
            weight=weight,
            start_percent=start_percent,
            end_percent=end_percent,
        )
        log.info("=" * 60)
        log.info("[KsanaEnhanceAVideoNode] CREATE_CONFIG CALLED - Enhance-A-Video (FETA)")
        log.info(f"  weight: {weight}")
        log.info(f"  start_percent: {start_percent}, end_percent: {end_percent}")
        log.info(f"  Created config: {config}")
        log.info("=" * 60)
        return (config,)


class KsanaExperimentalArgsNode:
    """
    Experimental Sampling Arguments Node.

    This node provides access to various experimental sampling optimizations:

    - CFG-Zero-Star: Optimizes CFG scaling to reduce oversaturation artifacts
    - FreSca: Frequency-domain filtering for cleaner CFG
    - TCFG: Tangential CFG to reduce color shifts
    - RAAG: Ratio-aware adaptive guidance
    - Bidirectional Sampling: Forward+backward temporal sampling
    - TSR: Temporal score rescaling for consistency
    """

    @classmethod
    def input_types(s):
        return {
            "required": {
                # CFG-Zero-Star: Optimized CFG scaling
                "cfg_zero_star": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable CFG-Zero-Star: Scales uncond to reduce oversaturation from high CFG. "
                        "Ref: https://github.com/WeichenFan/CFG-Zero-star",
                    },
                ),
                "use_zero_init": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When cfg_zero_star enabled, return zero noise for initial steps " "to stabilize sampling."
                        ),
                    },
                ),
                "zero_star_steps": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "Number of initial steps to apply zero initialization.",
                    },
                ),
                # FreSca: Frequency scaling
                "use_fresca": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable FreSca: Frequency-domain filtering to reduce CFG artifacts. "
                        "Ref: https://github.com/WikiChao/FreSca",
                    },
                ),
                "fresca_scale_low": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "FreSca low frequency scale factor.",
                    },
                ),
                "fresca_scale_high": (
                    "FLOAT",
                    {
                        "default": 1.25,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "FreSca high frequency scale factor.",
                    },
                ),
                "fresca_freq_cutoff": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 10000,
                        "tooltip": "FreSca frequency cutoff threshold.",
                    },
                ),
                # TCFG: Tangential CFG
                "use_tcfg": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable TCFG: Projects uncond onto tangent plane to reduce CFG color shifts. "
                        "Ref: https://arxiv.org/abs/2503.18137",
                    },
                ),
                # RAAG: Ratio-aware adaptive guidance
                "raag_alpha": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "RAAG alpha: Adaptively adjusts CFG based on cond/uncond ratio. 0 = disabled.",
                    },
                ),
                # Bidirectional sampling
                "bidirectional_sampling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable bidirectional temporal sampling (forward + backward). "
                        "Improves consistency but doubles compute. Ref: https://github.com/ff2416/WanFM",
                    },
                ),
                # TSR: Temporal score rescaling
                "temporal_score_rescaling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable TSR: Rescales noise pred based on temporal stats. "
                        "Ref: https://github.com/temporalscorerescaling/TSR",
                    },
                ),
                "tsr_k": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "TSR temperature. Lower = stronger rescaling.",
                    },
                ),
                "tsr_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "TSR sigma: How early TSR influences sampling.",
                    },
                ),
                # Attention split for multi-prompt
                "video_attention_split_steps": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated step indices for attention split with multiple prompts.",
                    },
                ),
            },
        }

    INPUT_TYPES = input_types  # Alias for ComfyUI compatibility

    RETURN_TYPES = (KSANA_EXPERIMENTAL_ARGS,)
    RETURN_NAMES = ("exp_args",)
    FUNCTION = "create_config"
    CATEGORY = "ksana/vace"
    DESCRIPTION = "Experimental sampling optimizations: CFG-Zero-Star, FreSca, TCFG, RAAG, Bidirectional, TSR."

    def create_config(
        self,
        cfg_zero_star: bool,
        use_zero_init: bool,
        zero_star_steps: int,
        use_fresca: bool,
        fresca_scale_low: float,
        fresca_scale_high: float,
        fresca_freq_cutoff: int,
        use_tcfg: bool,
        raag_alpha: float,
        bidirectional_sampling: bool,
        temporal_score_rescaling: bool,
        tsr_k: float,
        tsr_sigma: float,
        video_attention_split_steps: str,
    ):
        config = KsanaExperimentalConfig(
            cfg_zero_star=cfg_zero_star,
            use_zero_init=use_zero_init,
            zero_star_steps=zero_star_steps,
            use_fresca=use_fresca,
            fresca_scale_low=fresca_scale_low,
            fresca_scale_high=fresca_scale_high,
            fresca_freq_cutoff=fresca_freq_cutoff,
            use_tcfg=use_tcfg,
            raag_alpha=raag_alpha,
            bidirectional_sampling=bidirectional_sampling,
            temporal_score_rescaling=temporal_score_rescaling,
            tsr_k=tsr_k,
            tsr_sigma=tsr_sigma,
            video_attention_split_steps=video_attention_split_steps,
        )
        log.info("=" * 60)
        log.info("[KsanaExperimentalArgsNode] CREATE_CONFIG CALLED - Experimental Args")
        log.info(
            f"  cfg_zero_star: {cfg_zero_star}, use_zero_init: {use_zero_init}, zero_star_steps: {zero_star_steps}"
        )
        log.info(
            f"  use_fresca: {use_fresca}, fresca_scale_low: {fresca_scale_low}, fresca_scale_high: {fresca_scale_high}"
        )
        log.info(f"  fresca_freq_cutoff: {fresca_freq_cutoff}, use_tcfg: {use_tcfg}, raag_alpha: {raag_alpha}")
        log.info(
            f"  bidirectional_sampling: {bidirectional_sampling}, temporal_score_rescaling: {temporal_score_rescaling}"
        )
        log.info(
            f"  tsr_k: {tsr_k}, tsr_sigma: {tsr_sigma}, video_attention_split_steps: {video_attention_split_steps}"
        )
        log.info(f"  Created config: {config}")
        log.info("=" * 60)
        return (config,)


class KsanaVideoControlConfigNode:
    """
    Video Control Config Node.

    Combines multiple video control parameters into a single configuration:
    - SLG (Skip Layer Guidance): Skip uncond inference on specific blocks
    - FETA (Enhance-A-Video): Improve temporal consistency
    - Experimental: CFG-Zero-Star, FreSca, TCFG, TSR, etc.

    Also supports WanVideoWrapper compatible types (SLGARGS, FETAARGS, EXPERIMENTALARGS).
    """

    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {},
            "optional": {
                # KsanaDiT native types
                "slg_args": (
                    KSANA_SLG_ARGS,
                    {"tooltip": "Skip Layer Guidance args: skip uncond inference on specific blocks for speed."},
                ),
                "feta_args": (
                    KSANA_FETA_ARGS,
                    {"tooltip": "Enhance-A-Video (FETA) args: improve temporal consistency."},
                ),
                "experimental_args": (
                    KSANA_EXPERIMENTAL_ARGS,
                    {"tooltip": "Experimental args: CFG-Zero-Star, FreSca, TCFG, TSR, etc."},
                ),
                # WanVideoWrapper compatible types
                "wanvideo_slg_args": (
                    WANVIDEO_SLG_ARGS,
                    {"tooltip": "WanVideoWrapper SLG args (compatible input)."},
                ),
                "wanvideo_feta_args": (
                    WANVIDEO_FETA_ARGS,
                    {"tooltip": "WanVideoWrapper FETA args (compatible input)."},
                ),
                "wanvideo_exp_args": (
                    WANVIDEO_EXPERIMENTAL_ARGS,
                    {"tooltip": "WanVideoWrapper Experimental args (compatible input)."},
                ),
            },
        }

    RETURN_TYPES = (KSANA_VIDEO_CONTROL_CONFIG,)
    RETURN_NAMES = ("video_control_config",)
    FUNCTION = "create_config"
    CATEGORY = "ksana"
    DESCRIPTION = (
        "Combines video control parameters (SLG, FETA, Experimental) into a single configuration "
        "for use with KsanaGeneratorNode. Also accepts WanVideoWrapper nodes as inputs."
    )

    def _convert_slg_args(self, args):
        """Convert dict to KsanaSLGConfig if needed."""
        if args is None:
            return None
        if isinstance(args, KsanaSLGConfig):
            return args
        if isinstance(args, dict):
            return KsanaSLGConfig(
                blocks=args.get("blocks", []),
                start_percent=args.get("start_percent", 0.1),
                end_percent=args.get("end_percent", 1.0),
            )
        return args

    def _convert_feta_args(self, args):
        """Convert dict to KsanaFETAConfig if needed."""
        if args is None:
            return None
        if isinstance(args, KsanaFETAConfig):
            return args
        if isinstance(args, dict):
            return KsanaFETAConfig(
                weight=args.get("weight", 2.0),
                start_percent=args.get("start_percent", 0.0),
                end_percent=args.get("end_percent", 1.0),
            )
        return args

    def _convert_experimental_args(self, args):
        """Convert dict to KsanaExperimentalConfig if needed."""
        if args is None:
            return None
        if isinstance(args, KsanaExperimentalConfig):
            return args
        if isinstance(args, dict):
            return KsanaExperimentalConfig(
                cfg_zero_star=args.get("cfg_zero_star", False),
                use_zero_init=args.get("use_zero_init", False),
                zero_star_steps=args.get("zero_star_steps", 0),
                use_fresca=args.get("use_fresca", False),
                fresca_scale_low=args.get("fresca_scale_low", 1.0),
                fresca_scale_high=args.get("fresca_scale_high", 1.25),
                fresca_freq_cutoff=args.get("fresca_freq_cutoff", 20),
                use_tcfg=args.get("use_tcfg", False),
                raag_alpha=args.get("raag_alpha", 0.0),
                bidirectional_sampling=args.get("bidirectional_sampling", False),
                temporal_score_rescaling=args.get("temporal_score_rescaling", False),
                tsr_k=args.get("tsr_k", 0.95),
                tsr_sigma=args.get("tsr_sigma", 1.0),
                video_attention_split_steps=args.get("video_attention_split_steps", ""),
            )
        return args

    def create_config(
        self,
        slg_args=None,
        feta_args=None,
        experimental_args=None,
        wanvideo_slg_args=None,
        wanvideo_feta_args=None,
        wanvideo_exp_args=None,
    ):
        final_slg = self._convert_slg_args(slg_args or wanvideo_slg_args)
        final_feta = self._convert_feta_args(feta_args or wanvideo_feta_args)
        final_exp = self._convert_experimental_args(experimental_args or wanvideo_exp_args)

        config = {
            "slg_args": final_slg,
            "feta_args": final_feta,
            "experimental_args": final_exp,
        }
        log.info("=" * 60)
        log.info("[KsanaVideoControlConfigNode] CREATE_CONFIG CALLED")
        log.info(f"  slg_args: {final_slg}")
        log.info(f"  feta_args: {final_feta}")
        log.info(f"  experimental_args: {final_exp}")
        log.info("=" * 60)
        return (config,)
