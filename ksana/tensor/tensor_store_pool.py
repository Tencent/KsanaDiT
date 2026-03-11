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

from .tensor_store import KsanaTensorStore, TensorValue


class KsanaTensorStorePool:
    """管理所有中间 tensor，与 ModelPool 同级。

    Node 间通过 key 引用 tensor，避免 tensor 跨 Ray 边界序列化。
    生命周期由 Engine.tensor_scope() 管理，scope 结束时 clear。

    支持存储单个 ``torch.Tensor`` 或 ``list[torch.Tensor]``。
    """

    def __init__(self):
        self._stores: dict[str, KsanaTensorStore] = {}

    def put(self, key: str, tensor: TensorValue) -> KsanaTensorStore:
        """存入 tensor（或 list[Tensor]），覆盖同名 key。"""
        store = KsanaTensorStore(key, tensor)
        self._stores[key] = store
        return store

    def get(self, key: str) -> TensorValue | None:
        """读取 tensor 值，不存在返回 None。"""
        store = self._stores.get(key)
        return store.tensor if store is not None else None

    def get_store(self, key: str) -> KsanaTensorStore:
        """获取 KsanaTensorStore 对象，不存在则抛出 KeyError。"""
        if key not in self._stores:
            raise KeyError(f"KsanaTensorStore '{key}' not found. Available keys: {list(self._stores.keys())}")
        return self._stores[key]

    def remove(self, key: str) -> None:
        """移除并释放指定 key 的 tensor（支持 list[Tensor]）。"""
        store = self._stores.pop(key, None)
        if store is not None:
            if isinstance(store.tensor, list):
                for i in range(len(store.tensor)):
                    store.tensor[i] = None
            store.tensor = None

    def has(self, key: str) -> bool:
        return key in self._stores

    def clear(self) -> None:
        """释放所有 tensor 引用并清空池（支持 list[Tensor]）。"""
        for store in self._stores.values():
            if isinstance(store.tensor, list):
                for i in range(len(store.tensor)):
                    store.tensor[i] = None
            store.tensor = None
        self._stores.clear()

    def keys(self) -> list[str]:
        return list(self._stores.keys())

    def __len__(self) -> int:
        return len(self._stores)

    def __repr__(self) -> str:
        return f"KsanaTensorStorePool(keys={self.keys()})"
