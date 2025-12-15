import torch
import safetensors

from .logger import log
from pathlib import Path


def load_torch_file(ckpt, device=None):
    if device is None:
        device = torch.device("cpu")
    return safetensors.torch.load_file(ckpt, device=str(device))


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
        shard_dict = load_torch_file(str(file_path), device=device)
        state_dict.update(shard_dict)

    return state_dict


def load_torch_files(file_list, device=None):
    state_dict = {}
    for file_path in file_list:
        log.info(f"Loading {file_path}...")
        shard_dict = load_torch_file(str(file_path), device=device)
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
