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

"""Generator tensor 操作函数 — 无状态工具函数。"""

import torch


def split_tensors(
    tensor: torch.Tensor | tuple[torch.Tensor] | list[torch.Tensor],
    start: int,
    end: int,
):
    """按 batch 维度切片 tensor（支持 Tensor / tuple / list）。"""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor[start:end]
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)([t[start:end] for t in tensor])
    else:
        raise ValueError("tensor must be torch.Tensor or tuple/list of torch.Tensor")


def cast_to(src: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """将 tensor 转换到目标 dtype 和 device。"""
    if src.dtype != dtype:
        src = src.to(dtype)
    if src.device != device:
        src = src.to(device)
    return src
