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
from .load import batch_safetensors_by_size, load_file_to_state_dict, load_files_to_state_dict
from .logger import log
from .profile import time_range


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


def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha


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
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
    strength: float = 1.0,
):
    is_native_weight = any("diffusion_model." in key for key in lora_sd)
    merged_cnt = 0
    for key, value in model_sd.items():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_sd:
            lora_down = lora_sd[lora_down_name]
            lora_up = lora_sd[lora_up_name]

            if lora_alpha_name in lora_sd:
                rank = lora_down.shape[0]
                lora_alpha = float(lora_sd[lora_alpha_name])
                scaling_factor = lora_alpha / rank
            else:
                scaling_factor = 1.0

            delta_w_ = strength * scaling_factor * torch.matmul(lora_up, lora_down)

            # Reuse tensor cache to reduce memory usage
            temp = torch.empty_like(value, dtype=torch.float32)
            if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                # FP8 类型需要先转换为 float32 再计算
                scale = get_weight_scale(model_sd, key, device=value.device)
                temp.copy_(value.data).mul_(scale.to(dtype=torch.float32)).add_(delta_w_)
                value.data.copy_(temp.to(run_dtype))
            else:
                temp.copy_(value.data).add_(delta_w_)
                value.data.copy_(temp.to(run_dtype))
            merged_cnt += 1
    log.info(f"merged {merged_cnt} lora weights")
    return model_sd


@time_range
def load_state_dict_and_merge_lora(
    model_path: str, loras_list: list = None, run_dtype: torch.dtype = None, device=None, vace_model: str = None
):
    sd = {}

    if loras_list is not None and run_dtype is None:
        raise RuntimeError("run_dtype cannot be None when loras_list is provided.")
    need_merge = loras_list is not None and len(loras_list) > 0
    if not need_merge:
        device = "cpu"
        loras_list = []

    # TODO(rockcao): support merge lora on gpu
    device = "cpu"

    log.info(f"load_state_dict_and_merge_lora on rank {get_rank_id()} via device {device}")

    if Path(model_path).is_file():
        files_list = [[model_path]]
    elif Path(model_path).is_dir():
        # group files by size to reduce memory usage
        files_list = batch_safetensors_by_size(model_path)
    else:
        raise ValueError(f"model_path {model_path} is not a file or dir")

    for files in files_list:
        base_sd = load_files_to_state_dict(files, device=device)

        for lora in loras_list:
            log.info(f"start to merge lora: {lora.path}")
            lora_sd = load_file_to_state_dict(lora.path, device=device)
            base_sd = merge_lora_weight(base_sd, lora_sd, run_dtype, strength=lora.strength)
            del lora_sd

        log.debug("start to offload to cpu")
        for _, value in base_sd.items():
            value.data = value.to("cpu")
        sd.update(base_sd)

    if vace_model:
        sd.update(load_file_to_state_dict(vace_model, device=device))
        log.info(f"loaded vace_model: {vace_model}")
    return sd
