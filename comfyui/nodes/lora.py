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

import folder_paths

from ksana.nodes import KSANA_CATEGORY_LORA, KSANA_LORA, build_list_of_lora_config


class KsanaLoraSelectMultiNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        lora_files = folder_paths.get_filename_list("loras")
        lora_files = ["Empty"] + lora_files  # Add "Empty" as the first option
        return {
            "required": {
                "lora_0": (lora_files, {"default": "Empty"}),
                "strength_0": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
                "lora_1": (lora_files, {"default": "Empty"}),
                "strength_1": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
                "lora_2": (lora_files, {"default": "Empty"}),
                "strength_2": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
                "lora_3": (lora_files, {"default": "Empty"}),
                "strength_3": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
                "lora_4": (lora_files, {"default": "Empty"}),
                "strength_4": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
            },
        }

    RETURN_TYPES = (KSANA_LORA,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_lora_path"
    CATEGORY = KSANA_CATEGORY_LORA
    DESCRIPTION = "Select a LoRA model from loras"

    def get_lora_path(
        self, lora_0, strength_0, lora_1, strength_1, lora_2, strength_2, lora_3, strength_3, lora_4, strength_4
    ):
        lora_inputs = [
            (lora_0, strength_0),
            (lora_1, strength_1),
            (lora_2, strength_2),
            (lora_3, strength_3),
            (lora_4, strength_4),
        ]
        return (build_list_of_lora_config(lora_inputs),)


class KsanaLoraSelectNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "lora": (["Empty"] + folder_paths.get_filename_list("loras"), {"default": "Empty"}),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "LORA strength, set to 0.0 to unmerge the LORA",
                    },
                ),
            },
        }

    RETURN_TYPES = (KSANA_LORA,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_lora_path"
    CATEGORY = KSANA_CATEGORY_LORA
    DESCRIPTION = "Select a LoRA model from loras"

    def get_lora_path(self, lora, strength):
        if lora and lora != "Empty":
            lora_path = folder_paths.get_full_path_or_raise("loras", lora)
        else:
            lora_path = None
        return (build_list_of_lora_config([(lora_path, strength)]),)


class KsanaLoraCombineNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "lora": (KSANA_LORA,),
            },
            "optional": {
                "low_noise_model_lora": (KSANA_LORA, {"default": None}),
            },
        }

    RETURN_TYPES = (KSANA_LORA,)
    RETURN_NAMES = ("lora",)
    FUNCTION = "combine_loras"
    CATEGORY = KSANA_CATEGORY_LORA
    DESCRIPTION = "Combine LoRAs for 2 models"

    def combine_loras(self, lora, low_noise_model_lora=None):
        combined_loras = [lora, low_noise_model_lora] if low_noise_model_lora is not None else [lora]
        return (combined_loras,)
