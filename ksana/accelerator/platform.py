import shutil
from functools import cache


@cache
def is_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


@cache
def is_npu() -> bool:
    return shutil.which("npu-smi") is not None
