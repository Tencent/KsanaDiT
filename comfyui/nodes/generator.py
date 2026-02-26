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

import ksana.nodes as nodes
from ksana.config.sample_config import KsanaSolverType
from ksana.nodes import (
    KSANA_CACHE_CONFIG,
    KSANA_DIFFUSION_MODEL,
    KSANA_GENERATE_OUTPUT,
    KSANA_TEXT_ENCODE_OUTPUT,
    KSANA_VACE_EMBEDS,
    KSANA_VAE_ENCODE_OUTPUT,
    KSANA_VIDEO_CONTROL_CONFIG,
)

import comfy
import comfy.model_management as mm
from comfy.utils import ProgressBar


class KsanaGeneratorNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "model": (
                    KSANA_DIFFUSION_MODEL,
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "positive": (
                    KSANA_TEXT_ENCODE_OUTPUT,
                    {"tooltip": "The conditioning describing the attributes you want to include in the image."},
                ),
                "negative": (
                    KSANA_TEXT_ENCODE_OUTPUT,
                    {"tooltip": "The conditioning describing the attributes you want to exclude from the image."},
                ),
                "image_embeds": (KSANA_VAE_ENCODE_OUTPUT, {"tooltip": "The latent image to denoise."}),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        # "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
                    },
                ),
                "solver_name": (
                    KsanaSolverType.get_supported_list(),
                    {
                        "default": KsanaSolverType.UNI_PC.value,
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, \
                            and style of the generated output.",
                    },
                ),
                "sample_guide_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the \
                            prompt. Higher values result in images more closely matching the prompt \
                            however too high values will negatively impact quality.",
                    },
                ),
                "sample_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": -1.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Noise schedule shift. For Qwen-Image, use -1 for auto (let pipeline compute).",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of \
                            the initial image allowing for image to image sampling.",
                    },
                ),
            },
            "optional": {
                "rope_function": (
                    ["default", "comfy"],
                    {
                        "default": "default",
                        "tooltip": "Select the rotary positional embedding implementation.",
                    },
                ),
                "low_sample_guide_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the \
                            prompt. Higher values result in images more closely matching the prompt \
                            however too high values will negatively impact quality.",
                    },
                ),
                "cache_config": (
                    KSANA_CACHE_CONFIG,
                    {"tooltip": "The cache configs."},
                ),
                "sigmas": ("FLOAT", {"forceInput": True}),
                "latent": (KSANA_VAE_ENCODE_OUTPUT, {"tooltip": "init Latents to use for video2video process"}),
                "add_noise_to_latent": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Add noise to the latent before sampling, needed "
                        "for video2video sampling when starting from clean video",
                    },
                ),
                "video_control_config": (
                    KSANA_VIDEO_CONTROL_CONFIG,
                    {"tooltip": "Video control config from KsanaVideoControlConfigNode (SLG, FETA, Experimental)."},
                ),
                "vace_embeds": (
                    KSANA_VACE_EMBEDS,
                    {"tooltip": "VACE embeddings from KsanaWanVaceToVideoNode for video-to-video control."},
                ),
            },
        }

    RETURN_TYPES = (KSANA_GENERATE_OUTPUT,)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    CATEGORY = "ksana"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def run(self, *args, **kwargs):
        return (
            nodes.generate(
                *args,
                comfy_device=mm.get_torch_device(),
                comfy_progress_bar_func=ProgressBar,
                comfy_free_mem_func=mm.free_memory,
                **kwargs,
            ),
        )
