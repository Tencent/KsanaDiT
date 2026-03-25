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

"""TensorValue / TensorPool 单元测试。

TensorPool 同时支持 TensorPoolKey（DAG 模式标准 key）和裸 TensorKey。
裸 TensorKey 会被 _normalize_key() 自动转换为 TensorPoolKey(0, key)，
node_id=0 是"外部注入"约定，与 DAG 节点的 node_id >= 1 不冲突。
"""

import pytest
import torch

from kdit.tensor import TensorKey, TensorPool, TensorValue
from kdit.tensor.tensor_pool_key import TensorPoolKey

# ── 辅助：构建 TensorPoolKey ────────────────────────────────────────────────

# 使用固定 node_id 构造 TensorPoolKey，模拟 DAG 中不同节点的输出
_K = lambda tk: TensorPoolKey(0, tk)  # noqa: E731


# ── TensorValue ─────────────────────────────────────────────────────────────


class TestTensorValue:
    def test_single_tensor(self):
        t = torch.zeros(2, 3)
        tensor_value = TensorValue(t)
        assert tensor_value.data is t
        assert not tensor_value.is_released

    def test_release_single(self):
        t = torch.zeros(2, 3)
        tensor_value = TensorValue(t)
        tensor_value.release()
        assert tensor_value.is_released
        assert tensor_value.data is None

    def test_release_idempotent(self):
        tensor_value = TensorValue(torch.ones(4))
        tensor_value.release()
        tensor_value.release()  # 第二次不应抛异常
        assert tensor_value.is_released

    def test_list_tensor(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        tensor_value = TensorValue(tensors)
        assert tensor_value.data is tensors
        assert not tensor_value.is_released

    def test_release_list(self):
        t0, t1 = torch.zeros(2), torch.ones(3)
        tensors = [t0, t1]
        tensor_value = TensorValue(tensors)
        tensor_value.release()
        assert tensor_value.is_released
        assert tensor_value.data is None

    def test_repr_single(self):
        tensor_value = TensorValue(torch.ones(4))
        r = repr(tensor_value)
        assert "torch.float32" in r
        assert "(4,)" in r

    def test_repr_list(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        tensor_value = TensorValue(tensors)
        r = repr(tensor_value)
        assert "list_len=2" in r
        assert "(2, 3)" in r
        assert "(4, 5)" in r

    def test_repr_released(self):
        tensor_value = TensorValue(torch.zeros(1))
        tensor_value.release()
        assert "released" in repr(tensor_value)


# ── TensorPool ─────────────────────────────────────────────────────────────


class TestTensorPool:
    def test_put_get(self):
        pool = TensorPool()
        t = torch.randn(3, 4)
        pool.put(_K(TensorKey.LATENTS), t)
        assert pool.has(_K(TensorKey.LATENTS))
        tensor_value = pool.get(_K(TensorKey.LATENTS))
        assert isinstance(tensor_value, TensorValue)
        assert tensor_value.data is t

    def test_get_missing_returns_none(self):
        pool = TensorPool()
        assert pool.get(_K(TensorKey.VIDEO)) is None

    def test_clear(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.POSITIVE), torch.zeros(1))
        pool.put(_K(TensorKey.NEGATIVE), torch.zeros(2))
        pool.put(_K(TensorKey.LATENTS), torch.zeros(3))
        assert len(pool) == 3
        pool.clear()
        assert len(pool) == 0
        assert pool.keys() == []

    def test_clear_releases_tensor_values(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), torch.zeros(4))
        tensor_value = pool.get(_K(TensorKey.LATENTS))
        pool.clear()
        assert tensor_value.is_released

    def test_clear_with_exclude(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.POSITIVE), torch.zeros(1))
        pool.put(_K(TensorKey.BASE_LATENT), torch.zeros(2))
        pool.put(_K(TensorKey.LATENTS), torch.zeros(3))
        tv_keep = pool.get(_K(TensorKey.BASE_LATENT))
        pool.clear(exclude=[_K(TensorKey.BASE_LATENT)])
        assert len(pool) == 1
        assert pool.has(_K(TensorKey.BASE_LATENT))
        assert not pool.has(_K(TensorKey.POSITIVE))
        assert not pool.has(_K(TensorKey.LATENTS))
        assert not tv_keep.is_released  # 保留的不被 release

    def test_overwrite_releases_old(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), torch.tensor(1.0))
        pool.put(_K(TensorKey.LATENTS), torch.tensor(2.0))
        new_tv = pool.get(_K(TensorKey.LATENTS))
        assert new_tv.data.item() == 2.0
        assert len(pool) == 1

    def test_keys(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.POSITIVE), torch.zeros(1))
        pool.put(_K(TensorKey.NEGATIVE), torch.zeros(1))
        assert set(pool.keys()) == {_K(TensorKey.POSITIVE), _K(TensorKey.NEGATIVE)}

    def test_repr_contains_keys(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), torch.zeros(1))
        assert "latents" in repr(pool).lower()

    # ── list[Tensor] 支持 ──────────────────────────────────────────────

    def test_put_get_list_tensor(self):
        pool = TensorPool()
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        pool.put(_K(TensorKey.BASE_LATENT), tensors)
        assert pool.has(_K(TensorKey.BASE_LATENT))
        tensor_value = pool.get(_K(TensorKey.BASE_LATENT))
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2
        assert tensor_value.data[0] is tensors[0]
        assert tensor_value.data[1] is tensors[1]

    def test_overwrite_tensor_with_list(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), torch.tensor(1.0))
        pool.put(_K(TensorKey.LATENTS), [torch.tensor(2.0), torch.tensor(3.0)])
        tensor_value = pool.get(_K(TensorKey.LATENTS))
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2

    def test_overwrite_list_with_tensor(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), [torch.tensor(1.0), torch.tensor(2.0)])
        pool.put(_K(TensorKey.LATENTS), torch.tensor(3.0))
        tensor_value = pool.get(_K(TensorKey.LATENTS))
        assert isinstance(tensor_value.data, torch.Tensor)
        assert tensor_value.data.item() == 3.0

    def test_clear_with_list_tensors(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.POSITIVE), torch.zeros(1))
        pool.put(_K(TensorKey.BASE_LATENT), [torch.zeros(2), torch.ones(3)])
        assert len(pool) == 2
        pool.clear()
        assert len(pool) == 0

    def test_empty_list_tensor(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.BASE_LATENT), [])
        tensor_value = pool.get(_K(TensorKey.BASE_LATENT))
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 0

    def test_rename_basic(self):
        pool = TensorPool()
        t = torch.randn(3, 4)
        pool.put(_K(TensorKey.LATENTS), t)
        pool.rename(_K(TensorKey.LATENTS), _K(TensorKey.AUX_LATENT))
        assert not pool.has(_K(TensorKey.LATENTS))
        assert pool.has(_K(TensorKey.AUX_LATENT))
        assert torch.equal(pool.get(_K(TensorKey.AUX_LATENT)).data, t)

    def test_rename_overwrites_existing(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.LATENTS), torch.tensor(1.0))
        pool.put(_K(TensorKey.AUX_LATENT), torch.tensor(2.0))
        pool.rename(_K(TensorKey.LATENTS), _K(TensorKey.AUX_LATENT))
        assert not pool.has(_K(TensorKey.LATENTS))
        assert pool.get(_K(TensorKey.AUX_LATENT)).data.item() == 1.0

    def test_rename_missing_key_raises(self):
        pool = TensorPool()
        with pytest.raises(KeyError, match="old_key"):
            pool.rename(_K(TensorKey.LATENTS), _K(TensorKey.VIDEO))

    def test_rename_preserves_list_tensor(self):
        pool = TensorPool()
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        pool.put(_K(TensorKey.BASE_LATENT), tensors)
        pool.rename(_K(TensorKey.BASE_LATENT), _K(TensorKey.AUX_LATENT))
        assert not pool.has(_K(TensorKey.BASE_LATENT))
        tv = pool.get(_K(TensorKey.AUX_LATENT))
        assert isinstance(tv.data, list)
        assert len(tv.data) == 2
        assert torch.equal(tv.data[0], tensors[0])

    def test_rename_does_not_affect_other_keys(self):
        pool = TensorPool()
        pool.put(_K(TensorKey.POSITIVE), torch.zeros(1))
        pool.put(_K(TensorKey.LATENTS), torch.ones(2))
        pool.rename(_K(TensorKey.LATENTS), _K(TensorKey.VIDEO))
        assert pool.has(_K(TensorKey.POSITIVE))
        assert pool.has(_K(TensorKey.VIDEO))
        assert not pool.has(_K(TensorKey.LATENTS))
        assert len(pool) == 2


# ── 裸 TensorKey 自动转换 ──────────────────────────────────────────────────


class TestBareTensorKeyAutoNormalize:
    """裸 TensorKey 传入 TensorPool 时自动转换为 TensorPoolKey(0, key)。

    验证 ComfyUI adapter 等外部调用者无需手动包装 TensorPoolKey。
    """

    def test_put_with_bare_key_get_with_pool_key(self):
        """裸 TensorKey put → TensorPoolKey(0, key) get：同一条目。"""
        pool = TensorPool()
        t = torch.randn(3, 4)
        pool.put(TensorKey.LATENTS, t)  # 裸 key
        tv = pool.get(_K(TensorKey.LATENTS))  # TensorPoolKey
        assert tv is not None
        assert tv.data is t

    def test_put_with_pool_key_get_with_bare_key(self):
        """TensorPoolKey put → 裸 TensorKey get：同一条目。"""
        pool = TensorPool()
        t = torch.randn(2, 5)
        pool.put(_K(TensorKey.POSITIVE), t)
        tv = pool.get(TensorKey.POSITIVE)  # 裸 key
        assert tv is not None
        assert tv.data is t

    def test_has_with_bare_key(self):
        pool = TensorPool()
        pool.put(TensorKey.VIDEO, torch.zeros(1))
        assert pool.has(TensorKey.VIDEO)
        assert pool.has(_K(TensorKey.VIDEO))

    def test_clear_exclude_bare_key(self):
        """clear(exclude=[裸 TensorKey]) 正确保留对应条目。"""
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.LATENTS, torch.zeros(2))
        pool.clear(exclude=[TensorKey.LATENTS])  # 裸 key 作为 exclude
        assert len(pool) == 1
        assert pool.has(TensorKey.LATENTS)
        assert not pool.has(TensorKey.POSITIVE)

    def test_rename_bare_keys(self):
        """rename(裸 old, 裸 new) 正常工作。"""
        pool = TensorPool()
        t = torch.randn(4)
        pool.put(TensorKey.LATENTS, t)
        pool.rename(TensorKey.LATENTS, TensorKey.VIDEO)
        assert not pool.has(TensorKey.LATENTS)
        assert pool.has(TensorKey.VIDEO)
        assert pool.get(TensorKey.VIDEO).data is t

    def test_rename_mixed_keys(self):
        """rename(裸 old, TensorPoolKey new) 正常工作。"""
        pool = TensorPool()
        t = torch.randn(4)
        pool.put(TensorKey.LATENTS, t)
        pool.rename(TensorKey.LATENTS, _K(TensorKey.AUX_LATENT))
        assert not pool.has(TensorKey.LATENTS)
        assert pool.has(_K(TensorKey.AUX_LATENT))

    def test_no_deprecation_warning(self):
        """裸 TensorKey 不再触发 DeprecationWarning。"""
        import warnings

        pool = TensorPool()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            pool.put(TensorKey.LATENTS, torch.zeros(1))
            pool.get(TensorKey.LATENTS)
            pool.has(TensorKey.LATENTS)
            pool.clear(exclude=[TensorKey.LATENTS])
            pool.rename(TensorKey.LATENTS, TensorKey.VIDEO)
            pool.clear()


# ── TensorPool 引用计数 ─────────────────────────────────────────────────────


class TestTensorPoolRefCount:
    """TensorPool.register / consume / remove 引用计数机制。"""

    # 辅助：不同 node_id 的 TensorPoolKey
    _PK1 = TensorPoolKey(1, TensorKey.POSITIVE)
    _PK2 = TensorPoolKey(2, TensorKey.NEGATIVE)
    _PK3 = TensorPoolKey(3, TensorKey.LATENTS)

    def test_register_and_consume(self):
        """register 后 consume 递减引用计数，未归零时 tensor 仍在。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.register(self._PK1, ref_count=2)

        pool.consume(self._PK1)
        assert pool.has(self._PK1), "ref_count=1，tensor 应仍存在"

    def test_consume_auto_release(self):
        """引用计数归零时 tensor 自动释放。"""
        pool = TensorPool()
        t = torch.randn(4)
        pool.put(self._PK1, t)
        pool.register(self._PK1, ref_count=2)

        pool.consume(self._PK1)
        pool.consume(self._PK1)
        assert not pool.has(self._PK1), "ref_count=0，tensor 应已释放"
        assert len(pool) == 0

    def test_consume_unknown_key_noop(self):
        """consume 未注册的 key 静默跳过，不抛异常。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        # 未调用 register，直接 consume
        pool.consume(self._PK1)
        assert pool.has(self._PK1), "未注册 ref_count 的 tensor 不受 consume 影响"

    def test_remove(self):
        """remove 立即释放 tensor，不管引用计数。"""
        pool = TensorPool()
        t = torch.randn(4)
        pool.put(self._PK1, t)
        pool.register(self._PK1, ref_count=5)

        pool.remove(self._PK1)
        assert not pool.has(self._PK1)
        assert len(pool) == 0

    def test_remove_nonexistent_noop(self):
        """remove 不存在的 key 静默跳过。"""
        pool = TensorPool()
        pool.remove(self._PK1)  # 不抛异常

    def test_clear_resets_ref_counts(self):
        """clear 同时清理 tensor 和引用计数记录。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.put(self._PK2, torch.randn(4))
        pool.register(self._PK1, ref_count=3)
        pool.register(self._PK2, ref_count=1)

        pool.clear()
        assert len(pool) == 0
        # 再次 consume 不应抛异常（ref_counts 已清空）
        pool.consume(self._PK1)
        pool.consume(self._PK2)

    def test_clear_with_exclude_preserves_ref_count(self):
        """clear(exclude=[...]) 保留被排除 key 的引用计数。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.put(self._PK2, torch.randn(4))
        pool.register(self._PK1, ref_count=2)
        pool.register(self._PK2, ref_count=1)

        pool.clear(exclude=[self._PK1])
        assert pool.has(self._PK1), "被排除的 key 应保留"
        assert not pool.has(self._PK2), "未排除的 key 应被清理"
        # _PK1 的 ref_count 仍有效
        pool.consume(self._PK1)
        assert pool.has(self._PK1), "ref_count 从 2 降到 1，仍在"
        pool.consume(self._PK1)
        assert not pool.has(self._PK1), "ref_count 归零，自动释放"

    def test_multi_consumer(self):
        """多个下游消费者场景：ref_count=3，consume 3 次后释放。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.register(self._PK1, ref_count=3)

        for i in range(2):
            pool.consume(self._PK1)
            assert pool.has(self._PK1), f"第 {i + 1} 次 consume 后 tensor 应仍在"

        pool.consume(self._PK1)
        assert not pool.has(self._PK1), "第 3 次 consume 后 tensor 应释放"

    def test_rename_preserves_ref_count(self):
        """rename 后引用计数跟随新 key。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.register(self._PK1, ref_count=2)

        pool.rename(self._PK1, self._PK3)
        assert not pool.has(self._PK1)
        assert pool.has(self._PK3)

        # 用新 key consume
        pool.consume(self._PK3)
        assert pool.has(self._PK3), "ref_count 从 2 降到 1"
        pool.consume(self._PK3)
        assert not pool.has(self._PK3), "ref_count 归零，自动释放"

    def test_register_overwrite(self):
        """重复 register 覆盖旧的引用计数。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        pool.register(self._PK1, ref_count=1)
        pool.register(self._PK1, ref_count=3)

        pool.consume(self._PK1)
        assert pool.has(self._PK1), "覆盖后 ref_count=3，consume 1 次后仍在"

    def test_consume_releases_tensor_value(self):
        """consume 归零时 TensorValue.release() 被调用，data 置 None。"""
        pool = TensorPool()
        pool.put(self._PK1, torch.randn(4))
        tv = pool.get(self._PK1)
        assert tv.data is not None

        pool.register(self._PK1, ref_count=1)
        pool.consume(self._PK1)
        assert tv.data is None, "release() 应将 data 置为 None"
