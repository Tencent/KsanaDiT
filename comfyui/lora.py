import folder_paths


def build_loras_list(lora_inputs):
    loras_list = []
    for lora_name, strength in lora_inputs:
        s = round(strength, 4) if not isinstance(strength, list) else strength
        if not lora_name or lora_name == "Empty":
            continue
        loras_list.append(
            {
                "path": folder_paths.get_full_path_or_raise("loras", lora_name),
                "strength": s,
            }
        )
    return (loras_list,)


class KsanaLoraSelectMultiNode:
    @classmethod
    def INPUT_TYPES(s):
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

    RETURN_TYPES = ("KSANALORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_lora_path"
    CATEGORY = "ksana/lora"
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
        return build_loras_list(lora_inputs)


class KsanaLoraSelectNode:
    @classmethod
    def INPUT_TYPES(s):
        lora_files = folder_paths.get_filename_list("loras")
        lora_files = ["Empty"] + lora_files  # Add "Empty" as the first option
        return {
            "required": {
                "lora": (lora_files, {"default": "Empty"}),
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

    RETURN_TYPES = ("KSANALORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "get_lora_path"
    CATEGORY = "ksana/lora"
    DESCRIPTION = "Select a LoRA model from loras"

    def get_lora_path(self, lora, strength):
        return build_loras_list([(lora, strength)])


class KsanaLoraCombineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": ("KSANALORA",),
            },
            "optional": {
                "low_noise_model_lora": ("KSANALORA", {"default": None}),
            },
        }

    RETURN_TYPES = ("KSANALORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "combine_loras"
    CATEGORY = "ksana/lora"
    DESCRIPTION = "Combine LoRAs for 2 models"

    def combine_loras(self, lora, low_noise_model_lora=None):
        combined_loras = [lora, low_noise_model_lora] if low_noise_model_lora is not None else [lora]
        return (combined_loras,)
