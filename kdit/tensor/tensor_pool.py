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


import warnings

from .tensor_key import TensorKey
from .tensor_pool_key import TensorPoolKey
from .tensor_value import TensorData, TensorValue

_TENSOR_KEY_DEPRECATION_MSG = (
    "Passing TensorKey directly to TensorPool is deprecated, use TensorPoolKey instead. "
    "This will be removed when legacy Node adapters are cleaned up."
)


def _warn_if_legacy_tensor_key(key: TensorKey | TensorPoolKey) -> None:
    if isinstance(key, TensorKey):
        warnings.warn(_TENSOR_KEY_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


class TensorPool:
    """管理所有中间 tensor，与 ModelPool 同级。

    Node 间通过 ``TensorKey`` 引用 tensor，避免 tensor 跨 Ray 边界序列化。
    生命周期由 ``Engine.tensor_scope()`` 管理，scope 结束时 ``clear(exclude=keep)``。

    同时支持 ``TensorKey``（旧接口，已 deprecated）和 ``TensorPoolKey``（新 DAG 接口）作为 key。

    公开方法: ``put`` / ``get`` / ``clear`` / ``has`` / ``keys`` / ``__len__`` / ``__repr__``
    """

    def __init__(self):
        self._stores: dict[TensorKey | TensorPoolKey, TensorValue] = {}

    def put(self, key: TensorKey | TensorPoolKey, data: TensorData) -> None:
        """存入 tensor（或 list[Tensor]），覆盖同名 key。"""
        _warn_if_legacy_tensor_key(key)
        self._stores[key] = TensorValue(data)

    def get(self, key: TensorKey | TensorPoolKey) -> TensorValue | None:
        """读取 TensorValue，不存在返回 None。"""
        _warn_if_legacy_tensor_key(key)
        return self._stores.get(key)

    def has(self, key: TensorKey | TensorPoolKey) -> bool:
        """检查 key 是否存在于 pool 中。"""
        _warn_if_legacy_tensor_key(key)
        return key in self._stores

    def clear(self, exclude: list[TensorKey | TensorPoolKey] | None = None) -> None:
        """释放所有（或除 *exclude* 外的）tensor 引用并从池中移除。

        被排除的 key 保留在池中不被 release。
        """
        exclude_set = set(exclude) if exclude else set()
        keys_to_remove = [k for k in self._stores if k not in exclude_set]
        for k in keys_to_remove:
            self._stores[k].release()
            del self._stores[k]

    def rename(
        self,
        old_key: TensorKey | TensorPoolKey,
        new_key: TensorKey | TensorPoolKey,
    ) -> None:
        """将 old_key 重命名为 new_key，零拷贝。

        old_key 的 TensorValue 移到 new_key 下，old_key 从 pool 中删除。
        如果 new_key 已存在，直接覆盖（靠 Python 引用计数回收旧值）。

        Raises:
            KeyError: old_key 不存在于 pool 中。
        """
        _warn_if_legacy_tensor_key(old_key)
        _warn_if_legacy_tensor_key(new_key)
        if old_key == new_key:
            return
        if old_key not in self._stores:
            raise KeyError(f"TensorPool.rename: old_key {old_key!r} not found. Available keys: {self.keys()}")
        self._stores[new_key] = self._stores.pop(old_key)

    def keys(self) -> list[TensorKey | TensorPoolKey]:
        return list(self._stores.keys())

    def __len__(self) -> int:
        return len(self._stores)

    def __repr__(self) -> str:
        return f"TensorPool(keys={self.keys()})"
