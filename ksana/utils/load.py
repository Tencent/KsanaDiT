import os
from pathlib import Path

import safetensors
import torch

from .distribute import get_rank_id, is_multi_process
from .logger import log
from .prefetch import maybe_prefetch_file
from .profile import time_range


def _resolve_one_symlink_prefix(abs_path: str) -> str:
    """Resolve a single symlink prefix, without resolving through to CEPH.

    This is intended for CI where repo-local symlinks point into `/dockerdata/ci-models/...`.
    """

    parts = abs_path.split(os.sep)
    for i in range(len(parts), 1, -1):
        prefix = os.sep.join(parts[:i])
        if prefix and os.path.islink(prefix):
            target = os.readlink(prefix)
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(os.path.dirname(prefix), target))
            rest = os.sep.join(parts[i:])
            return os.path.join(target, rest) if rest else target
    return abs_path


def _map_ci_models_single_multi(path: str, root: str) -> str:
    rank = get_rank_id()
    multi = is_multi_process()

    token_multi = "/multi/"
    token_single = "/single/"

    if multi and token_single in path:
        prefix, suffix = path.split(token_single, 1)
        return prefix + token_multi + f"rank{rank}/" + suffix

    return path


def resolve_ci_models_ckpt_path(ckpt: str) -> str:
    root = os.getenv("KSANA_CI_MODELS_ROOT")
    if not root:
        return ckpt
    root = root.rstrip("/")

    path = os.path.abspath(ckpt)
    path = _resolve_one_symlink_prefix(path)

    # Only rewrite paths under the configured CI root.
    if not (path == root or path.startswith(root + "/")):
        raise RuntimeError(f"Path '{path}' is not within the CI models root '{root}'")

    return _map_ci_models_single_multi(path, root)


def remove_prefix_from_sd_inplace(state_dict: dict, prefix: str) -> dict:
    """
    Remove a specified prefix from all keys in the state dictionary (in-place).

    Args:
        state_dict (dict): The state dictionary with keys to be modified (modified in-place).
        prefix (str): The prefix string to be removed from the keys.

    Returns:
        dict: The same state dictionary with the specified prefix removed from the keys.
    """
    if not prefix:
        return state_dict

    keys_to_update = [key for key in state_dict if key.startswith(prefix)]
    if len(keys_to_update) > 0:
        log.info(f"{len(keys_to_update)} keys with multiple prefixes are to be deleted.")

    for old_key in keys_to_update:
        state_dict[old_key.removeprefix(prefix)] = state_dict.pop(old_key)

    return state_dict


def unet_prefix_from_state_dict(state_dict):
    # Note: candidates 里面内容的顺序不能随意调换
    candidates = [
        "model.diffusion_model.",  # qwen models
        "model.",  # wan models
    ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    return top


def remove_comfyui_prefix_from_state_dict(state_dict: dict) -> dict:
    detected_prefix = unet_prefix_from_state_dict(state_dict)
    return remove_prefix_from_sd_inplace(state_dict, detected_prefix)


def load_file_to_state_dict(ckpt, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt = resolve_ci_models_ckpt_path(str(ckpt))
    maybe_prefetch_file(ckpt)

    # 根据文件扩展名选择加载方式
    ckpt_path = Path(ckpt)

    if ckpt_path.suffix == ".safetensors":
        state_dict = safetensors.torch.load_file(ckpt, device=str(device))
    elif ckpt_path.suffix in [".pt", ".pth"]:
        state_dict = torch.load(ckpt, map_location=device)
    else:
        raise ValueError(f"Unsupported file format: {ckpt_path.suffix}. Supported formats: .safetensors, .pt, .pth")
    state_dict = remove_comfyui_prefix_from_state_dict(state_dict)
    return state_dict


def load_sharded_safetensors(model_dir, device=None):
    """
    加载目录下所有的 safetensors 文件

    Args:
        model_dir: 包含 .safetensors 文件的目录
        device: 目标设备

    Returns:
        合并后的 state_dict
    """
    model_dir = Path(model_dir)

    # 查找所有 safetensors 文件
    safetensors_files = sorted(model_dir.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in: {model_dir}")

    # 加载所有文件并合并
    state_dict = {}
    for file_path in safetensors_files:
        log.debug(f"Loading {file_path.name}...")
        shard_dict = load_file_to_state_dict(str(file_path), device=device)
        state_dict.update(shard_dict)

    return state_dict


def load_files_to_state_dict(file_list, device=None):
    state_dict = {}
    for file_path in file_list:
        log.info(f"Loading {file_path}...")
        shard_dict = load_file_to_state_dict(str(file_path), device=device)
        state_dict.update(shard_dict)

    return state_dict


def batch_safetensors_by_size(model_dir, max_batch_size_gb=32):
    """
    将 safetensors 文件按大小分组,每组不超过指定大小

    Args:
        model_dir: 包含 .safetensors 文件的目录
        max_batch_size_gb: 每组文件的最大大小(GB),默认32GB

    Returns:
        list of list: 分组后的文件列表,每个内部列表的文件总大小不超过 max_batch_size_gb
    """
    model_dir = Path(model_dir)

    # 查找所有 safetensors 文件并获取大小
    safetensors_files = sorted(model_dir.glob("*.safetensors"))

    if not safetensors_files:
        return []

    # 获取文件大小信息 [(file_path, size_in_bytes), ...]
    files_with_size = [(f, f.stat().st_size) for f in safetensors_files]

    # 按大小分组
    max_batch_size_bytes = max_batch_size_gb * 1024**3  # 转换为字节
    batches = []
    current_batch = []
    current_batch_size = 0

    for file_path, file_size in files_with_size:
        # 如果单个文件就超过限制,单独成组
        if file_size > max_batch_size_bytes:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            batches.append([file_path])
            continue

        # 如果加入当前文件会超过限制,开始新组
        if current_batch_size + file_size > max_batch_size_bytes:
            batches.append(current_batch)
            current_batch = [file_path]
            current_batch_size = file_size
        else:
            current_batch.append(file_path)
            current_batch_size += file_size

    # 添加最后一组
    if current_batch:
        batches.append(current_batch)

    return batches


@time_range
def load_state_dict(model, state_dict, assign=False, strict=False):
    load_result = model.load_state_dict(state_dict, strict=strict, assign=assign)
    if load_result.missing_keys:
        error_msg = (
            f"Runtime Error: Detected {len(load_result.missing_keys)} missing model parameters during weight loading."
        )
        raise RuntimeError(error_msg)
    if load_result.unexpected_keys:
        log.warning(f"Detected {len(load_result.unexpected_keys)} unexpected keys")
    return load_result
