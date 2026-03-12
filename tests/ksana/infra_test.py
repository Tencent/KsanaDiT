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

"""V5 重构基础设施单元测试 — TensorValue/Pool, DistributedGroup, Node 层。"""

import pytest
import torch

from ksana.executor.distributed_group import DistributedGroupManager
from ksana.nodes.core.device_context import KsanaDeviceContext
from ksana.nodes.core.node_context import KsanaNodeContext
from ksana.nodes.core.node_factory import KsanaInferNodeFactory, KsanaLoaderNodeFactory
from ksana.nodes.core.node_types import KsanaDispatchPolicy, KsanaInferNodeType
from ksana.tensor import TensorKey, TensorPool, TensorValue

# ── TensorValue ─────────────────────────────────────────────────────────────


class TestTensorValue:
    def test_single_tensor(self):
        t = torch.zeros(2, 3)
        tv = TensorValue(t)
        assert tv.data is t
        assert not tv.is_released

    def test_release_single(self):
        t = torch.zeros(2, 3)
        tv = TensorValue(t)
        tv.release()
        assert tv.is_released
        assert tv.data is None

    def test_release_idempotent(self):
        tv = TensorValue(torch.ones(4))
        tv.release()
        tv.release()  # 第二次不应抛异常
        assert tv.is_released

    def test_list_tensor(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        tv = TensorValue(tensors)
        assert tv.data is tensors
        assert not tv.is_released

    def test_release_list(self):
        t0, t1 = torch.zeros(2), torch.ones(3)
        tensors = [t0, t1]
        tv = TensorValue(tensors)
        tv.release()
        assert tv.is_released
        assert tv.data is None

    def test_repr_single(self):
        tv = TensorValue(torch.ones(4))
        r = repr(tv)
        assert "torch.float32" in r
        assert "(4,)" in r

    def test_repr_list(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        tv = TensorValue(tensors)
        r = repr(tv)
        assert "list_len=2" in r
        assert "(2, 3)" in r
        assert "(4, 5)" in r

    def test_repr_released(self):
        tv = TensorValue(torch.zeros(1))
        tv.release()
        assert "released" in repr(tv)


# ── TensorPool ─────────────────────────────────────────────────────────────


class TestTensorPool:
    def test_put_get(self):
        pool = TensorPool()
        t = torch.randn(3, 4)
        pool.put(TensorKey.LATENTS, t)
        assert pool.has(TensorKey.LATENTS)
        tv = pool.get(TensorKey.LATENTS)
        assert isinstance(tv, TensorValue)
        assert tv.data is t

    def test_get_missing_returns_none(self):
        pool = TensorPool()
        assert pool.get(TensorKey.VIDEO) is None

    def test_clear(self):
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.NEGATIVE, torch.zeros(2))
        pool.put(TensorKey.LATENTS, torch.zeros(3))
        assert len(pool) == 3
        pool.clear()
        assert len(pool) == 0
        assert pool.keys() == []

    def test_clear_releases_tensor_values(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.zeros(4))
        tv = pool.get(TensorKey.LATENTS)
        pool.clear()
        assert tv.is_released

    def test_clear_with_exclude(self):
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.IMAGE_EMBEDS, torch.zeros(2))
        pool.put(TensorKey.LATENTS, torch.zeros(3))
        tv_keep = pool.get(TensorKey.IMAGE_EMBEDS)
        pool.clear(exclude=[TensorKey.IMAGE_EMBEDS])
        assert len(pool) == 1
        assert pool.has(TensorKey.IMAGE_EMBEDS)
        assert not pool.has(TensorKey.POSITIVE)
        assert not pool.has(TensorKey.LATENTS)
        assert not tv_keep.is_released  # 保留的不被 release

    def test_overwrite_releases_old(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.tensor(1.0))
        pool.put(TensorKey.LATENTS, torch.tensor(2.0))
        new_tv = pool.get(TensorKey.LATENTS)
        assert new_tv.data.item() == 2.0
        assert len(pool) == 1

    def test_keys(self):
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.NEGATIVE, torch.zeros(1))
        assert sorted(pool.keys()) == sorted([TensorKey.POSITIVE, TensorKey.NEGATIVE])

    def test_repr_contains_keys(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.zeros(1))
        assert "latents" in repr(pool).lower()

    # ── list[Tensor] 支持 ──────────────────────────────────────────────

    def test_put_get_list_tensor(self):
        pool = TensorPool()
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        pool.put(TensorKey.IMAGE_EMBEDS, tensors)
        assert pool.has(TensorKey.IMAGE_EMBEDS)
        tv = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tv.data, list)
        assert len(tv.data) == 2
        assert tv.data[0] is tensors[0]
        assert tv.data[1] is tensors[1]

    def test_overwrite_tensor_with_list(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.tensor(1.0))
        pool.put(TensorKey.LATENTS, [torch.tensor(2.0), torch.tensor(3.0)])
        tv = pool.get(TensorKey.LATENTS)
        assert isinstance(tv.data, list)
        assert len(tv.data) == 2

    def test_overwrite_list_with_tensor(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, [torch.tensor(1.0), torch.tensor(2.0)])
        pool.put(TensorKey.LATENTS, torch.tensor(3.0))
        tv = pool.get(TensorKey.LATENTS)
        assert isinstance(tv.data, torch.Tensor)
        assert tv.data.item() == 3.0

    def test_clear_with_list_tensors(self):
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.IMAGE_EMBEDS, [torch.zeros(2), torch.ones(3)])
        assert len(pool) == 2
        pool.clear()
        assert len(pool) == 0

    def test_empty_list_tensor(self):
        pool = TensorPool()
        pool.put(TensorKey.IMAGE_EMBEDS, [])
        tv = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tv.data, list)
        assert len(tv.data) == 0


# ── DistributedGroupManager ───────────────────────────────────────────


class TestDistributedGroupManager:
    def test_default_state(self):
        mgr = DistributedGroupManager()
        assert mgr.rank_id == 0
        assert mgr.world_size == 1
        assert not mgr.is_initialized

    def test_init_single_gpu(self):
        mgr = DistributedGroupManager()
        mgr.init(0, 1)
        # world_size=1 → 不算 initialized
        assert not mgr.is_initialized

    def test_broadcast_noop_when_not_initialized(self):
        mgr = DistributedGroupManager()
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(2))
        # 不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=[TensorKey.POSITIVE], src_rank=0)

    def test_broadcast_list_tensor_noop_when_not_initialized(self):
        mgr = DistributedGroupManager()
        pool = TensorPool()
        pool.put(TensorKey.IMAGE_EMBEDS, [torch.zeros(2, 3), torch.ones(4, 5)])
        # list[Tensor] 也不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=[TensorKey.IMAGE_EMBEDS], src_rank=0)
        # 验证 pool 中的值未被修改
        tv = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tv.data, list)
        assert len(tv.data) == 2


# ── KsanaDispatchPolicy ────────────────────────────────────────────────────


class TestDispatchPolicy:
    def test_three_values(self):
        assert len(KsanaDispatchPolicy) == 3
        assert KsanaDispatchPolicy.ALL_ALL_ALL is not None
        assert KsanaDispatchPolicy.R0_R0_BCAST is not None
        assert KsanaDispatchPolicy.ALL_R0_R0 is not None


# ── KsanaInferNodeType ────────────────────────────────────────────────


class TestKsanaInferNodeType:
    def test_all_types_exist(self):
        expected = {
            "TEXT_ENCODE",
            "VAE_ENCODE_SPATIAL",
            "VAE_ENCODE_IMAGES",
            "VAE_DECODE",
            "GENERATE",
        }
        actual = {nt.name for nt in KsanaInferNodeType}
        assert actual == expected


# ── DeviceContext ─────────────────────────────────────────────────────


class TestKsanaDeviceContext:
    def test_creation(self):
        ctx = KsanaDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=2,
        )
        assert ctx.rank_id == 0
        assert ctx.world_size == 2

    def test_frozen(self):
        ctx = KsanaDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )
        with pytest.raises(AttributeError):
            ctx.rank_id = 1


# ── KsanaNodeContext ────────────────────────────────────────────────────────


class TestNodeContext:
    def test_basic_creation(self):
        ctx = KsanaNodeContext(prompt="hello world")
        assert ctx.prompt == "hello world"
        assert ctx.metadata == {}

    def test_tensor_rejected(self):
        with pytest.raises(TypeError, match="Tensor"):
            KsanaNodeContext(prompt=torch.zeros(1))


# ── NodeFactory ─────────────────────────────────────────────────────


class TestNodeFactory:
    def test_loader_factory_has_entries(self):
        # 确保 import ksana.nodes 后注册了 loader
        import ksana.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(KsanaLoaderNodeFactory._registry) > 0

    def test_infer_factory_has_entries(self):
        import ksana.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(KsanaInferNodeFactory._registry) > 0
