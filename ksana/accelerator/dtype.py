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
