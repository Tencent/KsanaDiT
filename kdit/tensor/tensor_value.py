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

# tensor_pool 中存储的原始值类型：单个 Tensor 或 list[Tensor]（如 BASE_LATENT / AUX_LATENT）
TensorData = torch.Tensor | list[torch.Tensor]


class TensorValue:
    """tensor_pool 中的存储单元，持有 ``Tensor`` 或 ``list[Tensor]``。

    - ``data``: 持有的原始 tensor 数据
    - ``release()``: 释放所有 tensor 引用（list 内元素逐个置 None → list.clear → data 置 None）
    - ``is_released``: 是否已释放
    """

    __slots__ = ["data"]

    def __init__(self, data: TensorData):
        self.data = data

    def release(self) -> None:
        """释放持有的 tensor 引用。

        对 ``list[Tensor]``：逐个元素置 None → ``list.clear()`` → ``data`` 置 None。
        对单个 ``Tensor``：直接 ``data`` 置 None。
        幂等——多次调用安全。
        """
        if self.data is None:
            return
        if isinstance(self.data, list):
            for i in range(len(self.data)):
                self.data[i] = None
            self.data.clear()
        self.data = None

    @property
    def is_released(self) -> bool:
        return self.data is None

    def __repr__(self) -> str:
        if self.data is None:
            return "TensorValue(released)"
        if isinstance(self.data, list):
            shapes = [tuple(t.shape) if t is not None else None for t in self.data]
            dtype = self.data[0].dtype if self.data and self.data[0] is not None else None
            return f"TensorValue(list_len={len(self.data)}, shapes={shapes}, dtype={dtype})"
        shape = tuple(self.data.shape)
        dtype = self.data.dtype
        return f"TensorValue(shape={shape}, dtype={dtype})"
