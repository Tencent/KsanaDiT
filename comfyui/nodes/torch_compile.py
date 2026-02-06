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
from ksana.nodes import KSANA_CATEGORY_CONFIGS, KSANA_TORCH_COMPILE


class KsanaTorchCompileNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "backend": (["inductor", "cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (
                    ["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"],
                    {"default": "default"},
                ),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "compile_transformer_blocks_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Compile only the transformer blocks, usually enough and can make compilation "
                        + "faster and less error prone",
                    },
                ),
            },
            "optional": {
                "dynamo_cache_size_limit": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "torch._dynamo.config.recompile_limit",
                    },
                ),
                "dynamo_recompile_limit": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 1024,
                        "step": 1,
                        "tooltip": "torch._dynamo.config.recompile_limit",
                    },
                ),
                "force_parameter_static_shapes": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "torch._dynamo.config.force_parameter_static_shapes"},
                ),
                "allow_unmerged_lora_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Allow LoRA application to be compiled with torch.compile to avoid graph breaks,"
                        + " causes issues with some LoRAs, mostly dynamic ones",
                    },
                ),
            },
        }

    RETURN_TYPES = (KSANA_TORCH_COMPILE,)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "set_args"
    CATEGORY = KSANA_CATEGORY_CONFIGS
    DESCRIPTION = (
        "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted."
    )

    def set_args(self, *args, **kwargs):
        return (nodes.torch_compile_config(*args, **kwargs),)
