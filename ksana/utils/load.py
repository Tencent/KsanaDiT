import torch
import safetensors
import os

from .logger import log
from pathlib import Path
from .prefetch import maybe_prefetch_file
from .distribute import get_rank_id, is_multi_process


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


def load_torch_file(ckpt, device=None):
    if device is None:
        device = torch.device("cpu")
    ckpt = resolve_ci_models_ckpt_path(str(ckpt))
    maybe_prefetch_file(ckpt)
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
