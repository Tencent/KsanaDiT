from __future__ import annotations

import os
from enum import Enum, auto, unique
from pathlib import Path

from ..utils import any_key_in_str, is_file_or_dir

VAE = ["vae"]
WAN2_2 = ["wan2.2", "wan22", "wan2_2", "wan_2_2", "wan_2.2"]
WAN2_1 = ["wan2.1", "wan21", "wan2_1", "wan_2_1", "wan_2.1"]
WAN_PARAMS = ["14b", "a14b"]
QWEN_IMAGE = ["qwen-image", "qwen_image"]

# TODO: support "s2v", "ti2v"
X2V_TYPES = ["t2v", "i2v", "vace"]
X2I_TYPES = ["t2i", "i2i"]


@unique
class KsanaModelKey(Enum):
    """
    暴露在外部comfy层面的模型key, 与model type的区别在于
    用于在model pool中查找model本身
    """

    T5TextEncoder = auto()
    Qwen2VLTextEncoder = auto()
    QwenImageVAE = auto()
    VAE_WAN2_1 = auto()
    VAE_WAN2_2 = auto()

    # diffusion models key, as well as pipeline key
    QwenImage_T2I = auto()
    Wan2_2_T2V_14B = auto()
    Wan2_2_I2V_14B = auto()
    Wan2_2_TI2V_5B = auto()
    Wan2_1_VACE_14B = auto()

    def is_i2v_type(self) -> bool:
        return self in [KsanaModelKey.Wan2_2_I2V_14B]

    def is_vace_type(self) -> bool:
        return self in [KsanaModelKey.Wan2_1_VACE_14B]


def get_model_key_from_path(model_path: str | list[str]):
    if isinstance(model_path, str):
        if not is_file_or_dir(model_path):
            raise ValueError(f"model_path {model_path} is not exist, or not a file or directory")
    elif isinstance(model_path, (list, tuple)):
        for p in model_path:
            if not is_file_or_dir(p):
                raise ValueError(f"model_path {p} in {model_path} is not exist, or not a file or directory")
        model_path = model_path[0]
    else:
        raise ValueError(f"model_path {model_path} is not exist, or not a file or directory")
    file_name = Path(model_path).name.lower()
    if any_key_in_str(VAE, file_name) is not None:
        if (
            os.path.isfile(model_path)
            and any_key_in_str(QWEN_IMAGE, file_name) is not None
            and file_name.find("hf") == -1
        ):
            # Note: comfyui use wan2.1 to load qwen-image vae
            return KsanaModelKey.VAE_WAN2_1
        elif any_key_in_str(QWEN_IMAGE, file_name) is not None:
            return KsanaModelKey.QwenImageVAE
        elif any_key_in_str(WAN2_2, file_name) is not None:
            return KsanaModelKey.VAE_WAN2_2
        elif any_key_in_str(WAN2_1, file_name) is not None:
            return KsanaModelKey.VAE_WAN2_1
        else:
            raise RuntimeError(
                f"can not detect model_key from model_name:{file_name}, model_path:{model_path} "
                f"maybe not in support list { WAN2_2 + WAN2_1 + QWEN_IMAGE}"
            )
    else:
        if any_key_in_str(QWEN_IMAGE, file_name) is not None:
            # model_name = QWEN_IMAGE[0]
            # model_type = "t2i"
            # model_size = "20B"
            return KsanaModelKey.QwenImage_T2I
        elif any_key_in_str(WAN2_2, file_name) is not None:
            idx = any_key_in_str(X2V_TYPES, file_name)
            if idx is None:
                raise RuntimeError(f"can not detect model_type:{X2V_TYPES} from file_name:{file_name}")
            if X2V_TYPES[idx] == "t2v":
                if any_key_in_str(WAN_PARAMS, file_name) is not None:
                    return KsanaModelKey.Wan2_2_T2V_14B
                else:
                    raise RuntimeError(f"can not detect model_size:{WAN_PARAMS} from file_name:{file_name}")
            elif X2V_TYPES[idx] == "i2v":
                if any_key_in_str(WAN_PARAMS, file_name) is not None:
                    return KsanaModelKey.Wan2_2_I2V_14B
                else:
                    raise RuntimeError(f"can not detect model_size:{WAN_PARAMS} from file_name:{file_name}")
            elif X2V_TYPES[idx] == "vace":
                # VACE is for Wan2.1
                if any_key_in_str(WAN_PARAMS, file_name) is not None:
                    return KsanaModelKey.Wan2_1_VACE_14B
                else:
                    raise RuntimeError(f"can not detect model_size:{WAN_PARAMS} from file_name:{file_name}")
            else:
                raise NotImplementedError(f"task_type {X2V_TYPES[idx]} is not in supported list {X2V_TYPES} yet")

        elif any_key_in_str(WAN2_1, file_name) is not None:
            idx = any_key_in_str(X2V_TYPES, file_name)
            if idx is not None and X2V_TYPES[idx] == "vace":
                return KsanaModelKey.Wan2_1_VACE_14B
            raise NotImplementedError(f"wan2.1 of {file_name} is not supported yet!")
        else:
            raise RuntimeError(
                f"can not detect model_key from model_name:{file_name}, "
                f"maybe not in support list {WAN2_2 + WAN2_1 + QWEN_IMAGE}"
            )
