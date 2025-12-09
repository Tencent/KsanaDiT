# adapt from wan

import torch
import torch.nn as nn
import os

from .profile import time_range
from .load import load_sharded_safetensors, load_torch_file
from .logger import log
from .utils import is_dir


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
        for buffer_name, buffer in module.named_buffers(recurse=False):
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
            lora_alpha = float(lora_sd[lora_alpha_name])
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = strength * scaling_factor * torch.matmul(lora_up, lora_down)
            if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                # FP8 类型需要先转换为 float32 再计算
                scale = get_weight_scale(model_sd, key, device=value.device)
                value.data = (value.to(dtype=torch.float32) * scale.to(dtype=torch.float32)) + delta_W
            else:
                value.data = value.data + delta_W
            merged_cnt += 1
    log.info(f"merged {merged_cnt} lora weights")
    return model_sd


def build_loras_list(lora_path: str, strength=1.0):
    return [{"path": lora_path, "strength": strength}]


@time_range
def merge_lora(model_sd_or_path: dict | str, loras_list, load_device=None):
    if not isinstance(model_sd_or_path, dict):
        if not (os.path.isfile(model_sd_or_path) or is_dir(model_sd_or_path)):
            raise ValueError(f"model_path {model_sd_or_path} must be a file or dir")
        model_sd = (
            load_sharded_safetensors(f"{model_sd_or_path}")
            if is_dir(model_sd_or_path)
            else load_torch_file(model_sd_or_path, device=load_device)
        )
    else:
        model_sd = model_sd_or_path

    if loras_list is None:
        return model_sd
    for lora in loras_list:
        log.info(f"start to merge lora: {lora['path']}")
        lora_sd = load_torch_file(lora["path"], device=load_device)
        model_sd = merge_lora_weight(model_sd, lora_sd, strength=lora["strength"])
        del lora_sd
    return model_sd
