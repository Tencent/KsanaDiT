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


from __future__ import annotations

import torch

# tensor_pool 中存储的值类型：单个 Tensor 或 list[Tensor]（如 IMAGE_EMBEDS）
TensorValue = torch.Tensor | list[torch.Tensor]


class KsanaTensorStore:
    """单个 tensor 的存储单元，用于 KsanaTensorStorePool 中管理中间计算结果。

    支持存储单个 ``torch.Tensor`` 或 ``list[torch.Tensor]``（例如多 prompt 的 image_embeds）。
    """

    __slots__ = ["key", "tensor"]

    def __init__(self, key: str, tensor: TensorValue):
        self.key = key
        self.tensor = tensor

    def __repr__(self) -> str:
        if self.tensor is None:
            return f"KsanaTensorStore(key={self.key!r}, shape=None, dtype=None)"
        if isinstance(self.tensor, list):
            shapes = [tuple(t.shape) for t in self.tensor]
            dtype = self.tensor[0].dtype if self.tensor else None
            return f"KsanaTensorStore(key={self.key!r}, list_len={len(self.tensor)}, shapes={shapes}, dtype={dtype})"
        shape = tuple(self.tensor.shape)
        dtype = self.tensor.dtype
        return f"KsanaTensorStore(key={self.key!r}, shape={shape}, dtype={dtype})"
