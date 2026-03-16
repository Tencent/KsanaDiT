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

"""KsanaModelKey — 标识一个具体的模型类别。

KsanaModelPool 只接受此类型。路径推导函数 get_model_key_from_path() 也在本文件中。
"""

from __future__ import annotations

import os
from enum import Enum, auto, unique
from pathlib import Path

from ..utils import any_key_in_str, is_file_or_dir

__all__ = [
    "KsanaModelKey",
    "get_model_key_from_path",
]

VAE = ["vae"]
WAN2_2 = ["wan2.2", "wan22", "wan2_2", "wan_2_2", "wan_2.2"]
WAN2_1 = ["wan2.1", "wan21", "wan2_1", "wan_2_1", "wan_2.1"]
WAN_PARAMS = ["14b", "a14b"]
QWEN_IMAGE = ["qwen-image", "qwen_image"]
QWEN_IMAGE_EDIT = ["qwen-image-edit", "qwen_image_edit"]

# TODO: support "s2v", "ti2v"
X2V_TYPES = ["t2v", "i2v", "vace"]
X2I_TYPES = ["t2i", "i2i", "edit"]


@unique
class KsanaModelKey(Enum):
    """标识一个具体的模型类别。

    零依赖枚举 — 可被任意子包安全导入，不会触发 kdit/__init__.py 的重量级导入链。

    KsanaModelPool 只接受此类型。用于：
      - LoaderNodeFactory / InferNodeFactory 按模型注册 Node
      - GeneratorFactory 按模型注册 Generator
      - settings 配置映射
    """

    # Text Encoders
    T5TextEncoder = auto()
    Qwen2VLTextEncoder = auto()
    Qwen2VLTextEncoderMultimodal = auto()

    # VAE
    QwenImageVAE = auto()
    VAE_WAN2_1 = auto()
    VAE_WAN2_2 = auto()

    # Diffusion Models — 与 PipelineKey 同名，因为不同 pipeline 的权重不同
    Wan2_2_T2V_14B = auto()
    Wan2_2_I2V_14B = auto()
    Wan2_2_TI2V_5B = auto()
    Wan2_1_VACE_14B = auto()
    QwenImage_T2I = auto()
    QwenImage_Edit = auto()

    # TODO: remove is_i2v_type is_image_type
    def is_i2v_type(self) -> bool:
        return self in (KsanaModelKey.Wan2_2_I2V_14B,)

    def is_image_type(self) -> bool:
        return self in (KsanaModelKey.QwenImage_T2I, KsanaModelKey.QwenImage_Edit)


# ── 路径推导 ───────────────────────────────────────────────────────────


def _resolve_model_path(model_path: str | list[str]) -> str:
    """校验并规范化 model_path，返回单个路径字符串。"""
    if isinstance(model_path, str):
        if not is_file_or_dir(model_path):
            raise ValueError(f"model_path {model_path} is not exist, or not a file or directory")
        return model_path
    if isinstance(model_path, (list, tuple)):
        for p in model_path:
            if not is_file_or_dir(p):
                raise ValueError(f"model_path {p} in {model_path} is not exist, or not a file or directory")
        return model_path[0]
    raise ValueError(f"model_path {model_path} is not exist, or not a file or directory")


def get_model_key_from_path(model_path: str | list[str]) -> KsanaModelKey:
    """从路径推导 KsanaModelKey。

    统一返回 KsanaModelKey。调用方如需 PipelineKey，
    需自行通过 PipelineKey[model_key.name] 转换。
    """
    model_path = _resolve_model_path(model_path)
    file_name = Path(model_path).name.lower()

    if any_key_in_str(VAE, file_name) is not None:
        return _detect_vae_key(model_path, file_name)
    return _detect_diffusion_key(file_name)


def _detect_vae_key(model_path: str, file_name: str) -> KsanaModelKey:
    """从 VAE 路径推导 KsanaModelKey。"""
    if os.path.isfile(model_path) and any_key_in_str(QWEN_IMAGE, file_name) is not None and "hf" not in file_name:
        return KsanaModelKey.VAE_WAN2_1  # comfyui use wan2.1 to load qwen-image vae
    if any_key_in_str(QWEN_IMAGE, file_name) is not None:
        return KsanaModelKey.QwenImageVAE
    if any_key_in_str(WAN2_2, file_name) is not None:
        return KsanaModelKey.VAE_WAN2_2
    if any_key_in_str(WAN2_1, file_name) is not None:
        return KsanaModelKey.VAE_WAN2_1
    raise RuntimeError(
        f"can not detect model_key from model_name:{file_name}, model_path:{model_path} "
        f"maybe not in support list {WAN2_2 + WAN2_1 + QWEN_IMAGE}"
    )


def _detect_diffusion_key(file_name: str) -> KsanaModelKey:
    """从非 VAE 路径推导 KsanaModelKey（Diffusion 模型）。"""
    if any_key_in_str(QWEN_IMAGE_EDIT, file_name) is not None:
        return KsanaModelKey.QwenImage_Edit
    if any_key_in_str(QWEN_IMAGE, file_name) is not None:
        return KsanaModelKey.QwenImage_T2I

    if any_key_in_str(WAN2_2, file_name) is not None:
        return _detect_wan22_key(file_name)

    if any_key_in_str(WAN2_1, file_name) is not None:
        idx = any_key_in_str(X2V_TYPES, file_name)
        if idx is not None and X2V_TYPES[idx] == "vace":
            return KsanaModelKey.Wan2_1_VACE_14B
        raise NotImplementedError(f"wan2.1 of {file_name} is not supported yet!")

    raise RuntimeError(
        f"can not detect model_key from model_name:{file_name}, "
        f"maybe not in support list {WAN2_2 + WAN2_1 + QWEN_IMAGE}"
    )


def _detect_wan22_key(file_name: str) -> KsanaModelKey:
    """从 Wan2.2 路径推导 KsanaModelKey。"""
    idx = any_key_in_str(X2V_TYPES, file_name)
    if idx is None:
        raise RuntimeError(f"can not detect model_type:{X2V_TYPES} from file_name:{file_name}")

    task_type = X2V_TYPES[idx]
    _key_map = {
        "t2v": KsanaModelKey.Wan2_2_T2V_14B,
        "i2v": KsanaModelKey.Wan2_2_I2V_14B,
        "vace": KsanaModelKey.Wan2_1_VACE_14B,
    }

    if task_type not in _key_map:
        raise NotImplementedError(f"task_type {task_type} is not in supported list {X2V_TYPES} yet")
    if any_key_in_str(WAN_PARAMS, file_name) is None:
        raise RuntimeError(f"can not detect model_size:{WAN_PARAMS} from file_name:{file_name}")
    return _key_map[task_type]
