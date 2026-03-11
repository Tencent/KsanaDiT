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

"""V5 重构基础设施单元测试 — Phase 1 (KsanaTensorStore/Pool, DistributedGroup) + Phase 2 (Node 层)。"""

import pytest
import torch

from ksana.executor.distributed_group import DistributedGroupManager
from ksana.nodes.core.device_context import KsanaDeviceContext
from ksana.nodes.core.node_context import KsanaNodeContext
from ksana.nodes.core.node_factory import KsanaInferNodeFactory, KsanaLoaderNodeFactory
from ksana.nodes.core.node_types import KsanaDispatchPolicy, KsanaInferNodeType
from ksana.tensor import KsanaTensorStore, KsanaTensorStorePool

# ── KsanaTensorStore ────────────────────────────────────────────────────────


class TestKsanaTensorStore:
    def test_slots(self):
        t = torch.zeros(2, 3)
        store = KsanaTensorStore("latent", t)
        assert store.key == "latent"
        assert store.tensor is t

    def test_repr(self):
        store = KsanaTensorStore("x", torch.ones(4))
        assert "x" in repr(store)
        assert "torch.float32" in repr(store)
        assert "(4,)" in repr(store)

    def test_list_tensor_store(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        store = KsanaTensorStore("img", tensors)
        assert store.key == "img"
        assert store.tensor is tensors

    def test_list_tensor_repr(self):
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        store = KsanaTensorStore("img", tensors)
        r = repr(store)
        assert "img" in r
        assert "list_len=2" in r
        assert "(2, 3)" in r
        assert "(4, 5)" in r


# ── KsanaTensorStorePool ───────────────────────────────────────────────────


class TestKsanaTensorStorePool:
    def test_put_get(self):
        pool = KsanaTensorStorePool()
        t = torch.randn(3, 4)
        pool.put("enc_out", t)
        assert pool.has("enc_out")
        assert pool.get("enc_out") is t

    def test_get_existing(self):
        pool = KsanaTensorStorePool()
        t = torch.tensor([1.0, 2.0])
        pool.put("a", t)
        assert pool.get("a") is t

    def test_get_missing_returns_none(self):
        pool = KsanaTensorStorePool()
        assert pool.get("nonexistent") is None

    def test_get_store_missing_raises(self):
        pool = KsanaTensorStorePool()
        with pytest.raises(KeyError, match="not found"):
            pool.get_store("missing")

    def test_remove(self):
        pool = KsanaTensorStorePool()
        pool.put("tmp", torch.zeros(1))
        pool.remove("tmp")
        assert not pool.has("tmp")
        assert len(pool) == 0

    def test_remove_nonexistent_is_noop(self):
        pool = KsanaTensorStorePool()
        pool.remove("ghost")  # 不应抛异常

    def test_clear(self):
        pool = KsanaTensorStorePool()
        for i in range(5):
            pool.put(f"k{i}", torch.zeros(i + 1))
        assert len(pool) == 5
        pool.clear()
        assert len(pool) == 0
        assert pool.keys() == []

    def test_overwrite(self):
        pool = KsanaTensorStorePool()
        pool.put("x", torch.tensor(1.0))
        pool.put("x", torch.tensor(2.0))
        assert pool.get("x").item() == 2.0
        assert len(pool) == 1

    def test_keys(self):
        pool = KsanaTensorStorePool()
        pool.put("a", torch.zeros(1))
        pool.put("b", torch.zeros(1))
        assert sorted(pool.keys()) == ["a", "b"]

    def test_repr_contains_keys(self):
        pool = KsanaTensorStorePool()
        pool.put("latent", torch.zeros(1))
        assert "latent" in repr(pool)

    # ── list[Tensor] 支持 ──────────────────────────────────────────────

    def test_put_get_list_tensor(self):
        pool = KsanaTensorStorePool()
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        pool.put("image_embeds", tensors)
        assert pool.has("image_embeds")
        result = pool.get("image_embeds")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] is tensors[0]
        assert result[1] is tensors[1]

    def test_list_tensor_store_repr(self):
        pool = KsanaTensorStorePool()
        tensors = [torch.zeros(2, 3), torch.ones(4, 5)]
        pool.put("img", tensors)
        store = pool.get_store("img")
        r = repr(store)
        assert "list_len=2" in r
        assert "(2, 3)" in r

    def test_overwrite_tensor_with_list(self):
        pool = KsanaTensorStorePool()
        pool.put("x", torch.tensor(1.0))
        pool.put("x", [torch.tensor(2.0), torch.tensor(3.0)])
        result = pool.get("x")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_overwrite_list_with_tensor(self):
        pool = KsanaTensorStorePool()
        pool.put("x", [torch.tensor(1.0), torch.tensor(2.0)])
        pool.put("x", torch.tensor(3.0))
        result = pool.get("x")
        assert isinstance(result, torch.Tensor)
        assert result.item() == 3.0

    def test_remove_list_tensor(self):
        pool = KsanaTensorStorePool()
        pool.put("img", [torch.zeros(2), torch.ones(3)])
        pool.remove("img")
        assert not pool.has("img")
        assert pool.get("img") is None

    def test_clear_with_list_tensors(self):
        pool = KsanaTensorStorePool()
        pool.put("a", torch.zeros(1))
        pool.put("b", [torch.zeros(2), torch.ones(3)])
        assert len(pool) == 2
        pool.clear()
        assert len(pool) == 0

    def test_empty_list_tensor(self):
        pool = KsanaTensorStorePool()
        pool.put("empty", [])
        result = pool.get("empty")
        assert isinstance(result, list)
        assert len(result) == 0


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
        pool = KsanaTensorStorePool()
        pool.put("x", torch.zeros(2))
        # 不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=["x"], src_rank=0)

    def test_broadcast_list_tensor_noop_when_not_initialized(self):
        mgr = DistributedGroupManager()
        pool = KsanaTensorStorePool()
        pool.put("img", [torch.zeros(2, 3), torch.ones(4, 5)])
        # list[Tensor] 也不应抛异常，直接跳过
        mgr.broadcast_tensors(tensor_pool=pool, keys=["img"], src_rank=0)
        # 验证 pool 中的值未被修改
        result = pool.get("img")
        assert isinstance(result, list)
        assert len(result) == 2


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
        assert ctx.tensor_refs == {}
        assert ctx.metadata == {}

    def test_tensor_rejected(self):
        with pytest.raises(TypeError, match="Tensor"):
            KsanaNodeContext(prompt=torch.zeros(1))

    def test_tensor_refs(self):
        ctx = KsanaNodeContext(tensor_refs={"latent": "latent_key"})
        assert ctx.tensor_refs["latent"] == "latent_key"


# ── NodeFactory ─────────────────────────────────────────────────────


class TestNodeFactory:
    def test_loader_factory_has_entries(self):
        # 确保 import ksana.nodes 后注册了 loader
        import ksana.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(KsanaLoaderNodeFactory._registry) > 0

    def test_infer_factory_has_entries(self):
        import ksana.nodes  # pylint: disable=unused-import # noqa: F401

        assert len(KsanaInferNodeFactory._registry) > 0
