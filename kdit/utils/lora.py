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

# adapt from wan


from pathlib import Path

import torch
import torch.nn as nn

from .distribute import get_rank_id
from .load import get_safetensors_list, load_file_to_state_dict, load_files_to_state_dict
from .logger import log
from .profile import time_profile


def model_safe_downcast(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    keep_in_fp32_modules: list[str] | tuple[str, ...] | None = None,
    keep_in_fp32_parameters: list[str] | tuple[str, ...] | None = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Downcast model parameters and buffers to a specified dtype, while keeping certain modules/parameters in fp32.

    Args:
        model: The PyTorch model to downcast
        dtype: The target dtype to downcast to (default: torch.bfloat16)
        keep_in_fp32_modules: List of module names to keep in fp32, fuzzy matching is supported
        keep_in_fp32_parameters: List of parameter names to keep in fp32, exact matching is required
        verbose: Whether to print information.

    Returns:
        The downcast model (modified in-place)
    """
    keep_in_fp32_modules = list(keep_in_fp32_modules or [])
    keep_in_fp32_modules.extend(getattr(model, "_keep_in_fp32_modules", []))
    keep_in_fp32_parameters = keep_in_fp32_parameters or []

    for name, module in model.named_modules():
        # Skip if module is in keep_in_fp32_modules list
        if any(keep_name in name for keep_name in keep_in_fp32_modules):
            if verbose:
                print(f"Skipping {name} because it is in keep_in_fp32_modules")
            continue

        # Downcast parameters
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = f"{name}.{param_name}" if name else param_name
            if param is not None:
                if full_param_name in keep_in_fp32_parameters and verbose:
                    print(f"Skipping {full_param_name} because it is in keep_in_fp32_parameters")
                # if not any(keep_name in full_param_name for keep_name in keep_in_fp32_parameters):
                else:
                    param.data = param.data.to(dtype)

        # Downcast buffers
        for _, buffer in module.named_buffers(recurse=False):
            if buffer is not None:
                buffer.data = buffer.data.to(dtype)
    return model


_LORA_SUFFIX_PATTERNS = (
    # Comfy/PEFT regular format
    (".lora_down.weight", ".lora_up.weight"),
    # Diffusers format used by many Qwen LoRAs
    (".lora_A.weight", ".lora_B.weight"),
)


def _resolve_lora_pair(weight_key: str, lora_sd: dict, is_native_weight: bool):
    base = "diffusion_model." if is_native_weight else ""
    for down_suffix, up_suffix in _LORA_SUFFIX_PATTERNS:
        down_key = base + weight_key.replace(".weight", down_suffix)
        up_key = base + weight_key.replace(".weight", up_suffix)
        if down_key in lora_sd and up_key in lora_sd:
            alpha_key = None
            for alpha_suffix in (".alpha", ".lora_alpha"):
                candidate = base + weight_key.replace(".weight", alpha_suffix)
                if candidate in lora_sd:
                    alpha_key = candidate
                    break
            return down_key, up_key, alpha_key
    return None


def get_weight_scale(model_sd, weight_name: str, device=None):
    """
    Get the scale weight for a given weight name.

    Args:
        model_sd: Model state dict
        weight_name: Weight name (e.g., 'blocks.0.cross_attn.k.weight')
        device: Device to create the default tensor on

    Returns:
        Scale weight tensor if exists, otherwise a tensor with value 1.0
    """
    # 将 .weight 替换为 .scale_weight
    scale_weight_name = weight_name.replace(".weight", ".scale_weight")

    # 检查 model_sd 中是否存在 scale_weight
    if scale_weight_name in model_sd:
        return model_sd[scale_weight_name]
    else:
        # 如果不存在,返回默认值 1.0 的张量
        return torch.tensor(1.0, device=device)


def merge_lora_weight(
    model_sd: dict,
    lora_sd: dict,
    run_dtype: torch.dtype,
    strength: float = 1.0,
):
    if strength == 0:
        log.warning("lora strength is 0, skipping merge")
        return model_sd

    merged_cnt = 0
    is_native_weight = any("diffusion_model." in key for key in lora_sd)
    for key, value in model_sd.items():
        if not key.endswith(".weight"):
            continue
        resolved = _resolve_lora_pair(key, lora_sd, is_native_weight)
        if resolved is not None:
            lora_down_name, lora_up_name, lora_alpha_name = resolved
            lora_down = lora_sd[lora_down_name]
            lora_up = lora_sd[lora_up_name]

            if lora_alpha_name is not None:
                rank = lora_down.shape[0]
                lora_alpha = float(lora_sd[lora_alpha_name])
                scaling_factor = lora_alpha / rank
            else:
                scaling_factor = 1.0

            lora_down_f32 = lora_down.to(dtype=torch.float32)
            lora_up_f32 = lora_up.to(dtype=torch.float32)
            delta_w_ = strength * scaling_factor * torch.matmul(lora_up_f32, lora_down_f32)

            # Reuse tensor cache to reduce memory usage
            temp = torch.empty_like(value, dtype=torch.float32)
            if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                # FP8 权重：反量化→合并→按运行精度（如 bfloat16）存储
                # 必须替换字典条目，而非拷贝回 FP8 存储
                # 因 FP8 的 3 位尾数会在重量化时丢失微小的 LoRA 增量
                scale = get_weight_scale(model_sd, key, device=value.device)
                temp.copy_(value.data).mul_(scale.to(dtype=torch.float32)).add_(delta_w_)
                model_sd[key] = temp.to(run_dtype)
            else:
                temp.copy_(value.data).add_(delta_w_)
                model_sd[key] = temp.to(run_dtype)
            merged_cnt += 1

    if merged_cnt > 0:
        log.info(f"merged {merged_cnt} lora weights")
    else:
        sample_keys = list(lora_sd.keys())[:5]
        log.warning(
            f"merged 0 lora weights. "
            f"lora keys may be incompatible with model weights. sample_lora_keys={sample_keys}"
        )
    return model_sd


@time_profile
def load_state_dict_and_merge_lora(
    model_path: str, loras_list: list = None, run_dtype: torch.dtype = None, device=None, model_patch_path: str = None
):
    state_dict = {}

    if loras_list is not None and run_dtype is None:
        raise RuntimeError("run_dtype cannot be None when loras_list is provided.")
    need_merge = loras_list is not None and len(loras_list) > 0
    if not need_merge:
        loras_list = []

    # TODO(rockcao): support merge lora on gpu
    device = "cpu"

    log.info(f"load_state_dict_and_merge_lora on rank {get_rank_id()} via device {device}")

    if Path(model_path).is_file():
        files = [model_path]
    elif Path(model_path).is_dir():
        files = get_safetensors_list(model_path)
    else:
        raise ValueError(f"model_path {model_path} is not a file or dir")

    state_dict = load_files_to_state_dict(files, device=device)

    if model_patch_path:
        log.info(f"loading model_patch_path: {model_patch_path}")
        state_dict.update(load_file_to_state_dict(model_patch_path, device=device))

    for lora in loras_list:
        log.info(f"starting to merge lora: {lora.path}")
        lora_sd = load_file_to_state_dict(lora.path, device=device)
        merge_lora_weight(state_dict, lora_sd, run_dtype, strength=lora.strength)
        del lora_sd

    return state_dict
