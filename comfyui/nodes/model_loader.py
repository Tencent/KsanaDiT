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

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar

import ksana.nodes as nodes
from ksana.config import KsanaLinearBackend


class KsanaVaceModelSelectNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "vace_model": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                "vace_low_model": (
                    ["Empty"] + folder_paths.get_filename_list("diffusion_models"),
                    {"default": "Empty"},
                ),
            },
        }

    RETURN_TYPES = (nodes.KSANA_VACE_MODEL,)
    RETURN_NAMES = ("vace_model",)
    FUNCTION = "set_models"
    CATEGORY = "ksana"

    @classmethod
    def VALIDATE_INPUTS(s):  # pylint: disable=invalid-name
        return True

    def set_models(self, vace_model, **kwargs):
        vace_model_path = folder_paths.get_full_path("diffusion_models", vace_model)
        vace_model = [vace_model_path]

        vace_low_model = kwargs.pop("vace_low_model", None)
        if vace_low_model is not None and vace_low_model != "Empty":
            vace_model.append(folder_paths.get_full_path("diffusion_models", vace_low_model))
        return (vace_model,)


class KsanaModelLoaderNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                # attention_config dtype > linear_backend dtype > run_dtype
                "run_dtype": (
                    ["float16", "bfloat16"],
                    {"default": "float16"},
                    {"tooltip": "dtype of running model"},
                ),
                "rms_dtype": (
                    ["float", "half"],
                    {"default": "float"},
                    {"tooltip": "dtype for RMSNorm operations: float (fp32 precision) or half (fp16/bf16 precision)"},
                ),
                "linear_backend": (
                    KsanaLinearBackend.get_supported_list(),
                    {"default": KsanaLinearBackend.DEFAULT.value},
                    {"tooltip": "linear_backend default use linear dtype from model"},
                ),
                "attention_config": (
                    nodes.KSANA_ATTENTION_CONFIG,
                    {"default": None},
                    {"tooltip": "attention config"},
                ),
                "low_noise_model_name": (
                    ["Empty"] + folder_paths.get_filename_list("diffusion_models"),
                    {"default": "Empty"},
                ),
                "model_boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.001,
                        "tooltip": "The boundary value used for high and low timesteps.",
                    },
                ),
                "torch_compile_args": (nodes.KSANA_TORCH_COMPILE, {"default": None}),
                "lora": (nodes.KSANA_LORA, {"default": None}),
                "vace_model": (nodes.KSANA_VACE_MODEL, {"default": None}),
            },
        }

    RETURN_TYPES = (nodes.KSANA_DIFFUSION_MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ksana"

    @classmethod
    def VALIDATE_INPUTS(s):  # pylint: disable=invalid-name
        return True

    def load_model(self, model_name, **kwargs):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        low_noise_model_name = kwargs.pop("low_noise_model_name", None)
        kwargs["high_noise_model_path"] = folder_paths.get_full_path("diffusion_models", model_name)
        if low_noise_model_name is not None and low_noise_model_name != "Empty":
            kwargs["low_noise_model_path"] = folder_paths.get_full_path("diffusion_models", low_noise_model_name)
        return (nodes.KsanaNodeModelLoader.load(comfy_progress_bar_func=ProgressBar, **kwargs),)
