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

from kdit.executor.distributed_group import DistributedGroupManager
from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_factory import InferNodeFactory, LoaderNodeFactory
from kdit.nodes.core.node_types import InferNodeType, NodeDispatchPolicy
from kdit.tensor import TensorKey, TensorPool, TensorValue

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
        pool.put(TensorKey.LATENTS, t)
        assert pool.has(TensorKey.LATENTS)
        tensor_value = pool.get(TensorKey.LATENTS)
        assert isinstance(tensor_value, TensorValue)
        assert tensor_value.data is t

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
        tensor_value = pool.get(TensorKey.LATENTS)
        pool.clear()
        assert tensor_value.is_released

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
        tensor_value = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2
        assert tensor_value.data[0] is tensors[0]
        assert tensor_value.data[1] is tensors[1]

    def test_overwrite_tensor_with_list(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.tensor(1.0))
        pool.put(TensorKey.LATENTS, [torch.tensor(2.0), torch.tensor(3.0)])
        tensor_value = pool.get(TensorKey.LATENTS)
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2

    def test_overwrite_list_with_tensor(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, [torch.tensor(1.0), torch.tensor(2.0)])
        pool.put(TensorKey.LATENTS, torch.tensor(3.0))
        tensor_value = pool.get(TensorKey.LATENTS)
        assert isinstance(tensor_value.data, torch.Tensor)
        assert tensor_value.data.item() == 3.0

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
        tensor_value = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 0

    def test_rename_basic(self):
        pool = TensorPool()
        t = torch.randn(3, 4)
        pool.put(TensorKey.LATENTS, t)
        pool.rename(TensorKey.LATENTS, TensorKey.INPUT_LATENT)
        assert not pool.has(TensorKey.LATENTS)
        assert pool.has(TensorKey.INPUT_LATENT)
        assert torch.equal(pool.get(TensorKey.INPUT_LATENT).data, t)

    def test_rename_overwrites_existing(self):
        pool = TensorPool()
        pool.put(TensorKey.LATENTS, torch.tensor(1.0))
        pool.put(TensorKey.INPUT_LATENT, torch.tensor(2.0))
        pool.rename(TensorKey.LATENTS, TensorKey.INPUT_LATENT)
        assert not pool.has(TensorKey.LATENTS)
        assert pool.get(TensorKey.INPUT_LATENT).data.item() == 1.0

    def test_rename_missing_key_raises(self):
        pool = TensorPool()
        with pytest.raises(KeyError, match="old_key"):
            pool.rename(TensorKey.LATENTS, TensorKey.VIDEO)

    def test_rename_preserves_list_tensor(self):
        pool = TensorPool()
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        pool.put(TensorKey.IMAGE_EMBEDS, tensors)
        pool.rename(TensorKey.IMAGE_EMBEDS, TensorKey.INPUT_LATENT)
        assert not pool.has(TensorKey.IMAGE_EMBEDS)
        tv = pool.get(TensorKey.INPUT_LATENT)
        assert isinstance(tv.data, list)
        assert len(tv.data) == 2
        assert torch.equal(tv.data[0], tensors[0])

    def test_rename_does_not_affect_other_keys(self):
        pool = TensorPool()
        pool.put(TensorKey.POSITIVE, torch.zeros(1))
        pool.put(TensorKey.LATENTS, torch.ones(2))
        pool.rename(TensorKey.LATENTS, TensorKey.VIDEO)
        assert pool.has(TensorKey.POSITIVE)
        assert pool.has(TensorKey.VIDEO)
        assert not pool.has(TensorKey.LATENTS)
        assert len(pool) == 2


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
        tensor_value = pool.get(TensorKey.IMAGE_EMBEDS)
        assert isinstance(tensor_value.data, list)
        assert len(tensor_value.data) == 2


# ── NodeDispatchPolicy ────────────────────────────────────────────────────


class TestDispatchPolicy:
    def test_three_values(self):
        assert len(NodeDispatchPolicy) == 3
        assert NodeDispatchPolicy.ALL_ALL_ALL is not None
        assert NodeDispatchPolicy.R0_R0_BCAST is not None
        assert NodeDispatchPolicy.ALL_R0_R0 is not None


# ── InferNodeType ────────────────────────────────────────────────


class TestKsanaInferNodeType:
    def test_all_types_exist(self):
        expected = {
            "TEXT_ENCODE",
            "VAE_ENCODE_SPATIAL",
            "VAE_ENCODE_IMAGES",
            "VAE_DECODE",
            "GENERATE",
        }
        actual = {nt.name for nt in InferNodeType}
        assert actual == expected


# ── NodeDeviceContext ─────────────────────────────────────────────────────


class TestKsanaDeviceContext:
    def test_creation(self):
        ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=2,
        )
        assert ctx.rank_id == 0
        assert ctx.world_size == 2

    def test_frozen(self):
        ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )
        with pytest.raises(AttributeError):
            ctx.rank_id = 1


# ── NodeContext ────────────────────────────────────────────────────────


class TestNodeContext:
    def test_basic_creation(self):
        ctx = NodeContext(prompt="hello world")
        assert ctx.prompt == "hello world"
        assert ctx.metadata == {}

    def test_tensor_rejected(self):
        with pytest.raises(TypeError, match="Tensor"):
            NodeContext(prompt=torch.zeros(1))


# ── NodeFactory ─────────────────────────────────────────────────────


class TestNodeFactory:
    def test_loader_factory_has_entries(self):
        # 确保 import kdit.nodes 后注册了 loader
        import kdit.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(LoaderNodeFactory._registry) > 0

    def test_infer_factory_has_entries(self):
        import kdit.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(InferNodeFactory._registry) > 0
