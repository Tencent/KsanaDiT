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


from .tensor_key import TensorKey
from .tensor_pool_key import TensorPoolKey
from .tensor_value import TensorData, TensorValue

#: 外部直接操作 TensorPool 时（如 ComfyUI adapter 通过 Engine 桥接方法），
#: 裸 TensorKey 自动包装为 TensorPoolKey(node_id=0, ...)。
#: node_id=0 是约定的"外部注入"标识，与 DAG 节点的 node_id >= 1 不冲突。
_EXTERNAL_NODE_ID = 0


def _normalize_key(key: TensorKey | TensorPoolKey) -> TensorPoolKey:
    """裸 TensorKey 自动转换为 TensorPoolKey(0, key)，TensorPoolKey 原样返回。"""
    if isinstance(key, TensorKey):
        return TensorPoolKey(_EXTERNAL_NODE_ID, key)
    return key


class TensorPool:
    """管理所有中间 tensor — 存储 + 引用计数 + 自动释放。

    Node 间通过 ``TensorKey`` 引用 tensor，避免 tensor 跨 Ray 边界序列化。
    引用计数归零时自动释放 tensor；``clear()`` 可一次性清理全部或排除指定 key。

    同时支持 ``TensorKey``（旧接口）和 ``TensorPoolKey``（新 DAG 接口）作为 key。

    公开方法: ``put`` / ``get`` / ``has`` / ``register`` / ``consume`` / ``remove`` /
    ``clear`` / ``rename`` / ``keys`` / ``__len__`` / ``__repr__``
    """

    __slots__ = ("_stores", "_ref_counts")

    def __init__(self):
        self._stores: dict[TensorPoolKey, TensorValue] = {}
        self._ref_counts: dict[TensorPoolKey, int] = {}

    def put(self, key: TensorKey | TensorPoolKey, data: TensorData) -> None:
        """存入 tensor（或 list[Tensor]），覆盖同名 key。"""
        key = _normalize_key(key)
        self._stores[key] = TensorValue(data)

    def get(self, key: TensorKey | TensorPoolKey) -> TensorValue | None:
        """读取 TensorValue，不存在返回 None。"""
        key = _normalize_key(key)
        return self._stores.get(key)

    def has(self, key: TensorKey | TensorPoolKey) -> bool:
        """检查 key 是否存在于 pool 中。"""
        key = _normalize_key(key)
        return key in self._stores

    # ── 引用计数 ──────────────────────────────────────────────

    def register(self, pool_key: TensorPoolKey, ref_count: int) -> None:
        """注册 tensor 的下游引用计数。

        由 Executor 在 Node 执行后调用，为每个 output tensor 设置消费者数量。

        Args:
            pool_key: tensor 的 PoolKey（必须是 TensorPoolKey，不做 normalize）。
            ref_count: 下游消费者数量（>= 0）。
        """
        self._ref_counts[pool_key] = ref_count

    def consume(self, pool_key: TensorPoolKey) -> None:
        """消费一次引用。归零时自动释放 tensor 并移除引用计数记录。

        如果 *pool_key* 没有注册引用计数（如外部注入的 tensor），静默跳过。
        """
        if pool_key not in self._ref_counts:
            return
        self._ref_counts[pool_key] -= 1
        if self._ref_counts[pool_key] <= 0:
            self.remove(pool_key)
            del self._ref_counts[pool_key]

    def remove(self, key: TensorKey | TensorPoolKey) -> None:
        """释放并移除指定 key 的 tensor。不存在时静默跳过。"""
        key = _normalize_key(key)
        if key in self._stores:
            self._stores[key].release()
            del self._stores[key]

    # ── 批量操作 ──────────────────────────────────────────────

    def clear(self, exclude: list[TensorKey | TensorPoolKey] | None = None) -> None:
        """释放所有（或除 *exclude* 外的）tensor 引用并从池中移除。

        被排除的 key 保留在池中不被 release。同时清理对应的引用计数记录。
        """
        exclude_set = {_normalize_key(k) for k in exclude} if exclude else set()
        keys_to_remove = [k for k in self._stores if k not in exclude_set]
        for k in keys_to_remove:
            self._stores[k].release()
            del self._stores[k]
            self._ref_counts.pop(k, None)

    def rename(
        self,
        old_key: TensorKey | TensorPoolKey,
        new_key: TensorKey | TensorPoolKey,
    ) -> None:
        """将 old_key 重命名为 new_key，零拷贝。

        old_key 的 TensorValue 和引用计数一并迁移到 new_key 下。
        如果 new_key 已存在，直接覆盖（靠 Python 引用计数回收旧值）。

        Raises:
            KeyError: old_key 不存在于 pool 中。
        """
        old_key = _normalize_key(old_key)
        new_key = _normalize_key(new_key)
        if old_key == new_key:
            return
        if old_key not in self._stores:
            raise KeyError(f"TensorPool.rename: old_key {old_key!r} not found. Available keys: {self.keys()}")
        self._stores[new_key] = self._stores.pop(old_key)
        if old_key in self._ref_counts:
            self._ref_counts[new_key] = self._ref_counts.pop(old_key)

    # ── 查询 ──────────────────────────────────────────────────

    def keys(self) -> list[TensorKey | TensorPoolKey]:
        return list(self._stores.keys())

    def __len__(self) -> int:
        return len(self._stores)

    def __repr__(self) -> str:
        return f"TensorPool(keys={self.keys()})"
