import torch
import safetensors

from .logger import log
from pathlib import Path


def load_torch_file(ckpt, safe_load=True, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    # if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                    #     tensor = tensor.to(device=device, copy=True)
                    sd[k] = tensor
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(
                            message, ckpt
                        )
                    )
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(
                            message, ckpt
                        )
                    )
            raise e
    else:
        raise ValueError(f"Only safetensors files are supported, but got: {ckpt}")
    return (sd, metadata) if return_metadata else sd


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
