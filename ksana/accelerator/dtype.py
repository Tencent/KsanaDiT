import torch

from . import platform

_NPU_ALLOWED_DTYPES: set[torch.dtype] = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
}


def _map_dtype_for_npu(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.complex64
    if dtype == torch.float64:
        return torch.float32
    if dtype in _NPU_ALLOWED_DTYPES:
        return dtype
    raise ValueError(f"NPU 不支持 {dtype}")


def normalize_dtype_for_platform(dtype: torch.dtype) -> torch.dtype:
    if platform.is_npu():
        return _map_dtype_for_npu(dtype)
    return dtype


__all__ = ["normalize_dtype_for_platform"]
