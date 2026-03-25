# Copyright 2026 Tencent

import pickle

import pytest
import torch

from kdit.models.model_base import ModelBase
from kdit.models.model_key import ModelKey
from kdit.models.model_pool import ModelPool
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_types import InferNodeType, IONodeType
from kdit.nodes.core.pin_hub import PinHub
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor.tensor_key import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey


def _node_def(node_id, *, model_key=None, node_type=InferNodeType.TEXT_ENCODE):
    """构建一个最小 NodeDef。"""
    return NodeDef(node_id=node_id, node_type=node_type, model_key=model_key)


class _StubModel(ModelBase):
    """测试用的最小 ModelBase 子类。"""

    def __init__(self, model_key: ModelKey):
        super().__init__(model_key=model_key, default_settings=None)

    def to(self, *args, **kwargs):
        return self


class TestPinHub:
    """PinHub 沙箱化读写测试。"""

    def test_put_and_get_tensor(self):
        """put_tensor 写入后，下游 PinHub 通过映射可读回。"""
        tp = TensorPool()
        mp = ModelPool()

        # 上游 node_id=0 写入 LATENTS
        upstream = PinHub(node_def=_node_def(0), input_pins={}, tensor_pool=tp, model_pool=mp)
        data = torch.randn(1, 4, 8, 8)
        upstream.put_tensor(TensorKey.LATENTS, data)

        # 验证 pool 中有 TensorPoolKey(0, LATENTS)
        assert tp.has(TensorPoolKey(0, TensorKey.LATENTS))

        # 下游 node_id=1 映射 LATENTS -> 上游的 TensorPoolKey(0, LATENTS)
        downstream = PinHub(
            node_def=_node_def(1),
            input_pins={"tensor": {TensorKey.LATENTS: TensorPoolKey(0, TensorKey.LATENTS)}},
            tensor_pool=tp,
            model_pool=mp,
        )
        result = downstream.get_tensor(TensorKey.LATENTS)
        assert result is data

    def test_get_tensor_unmapped_returns_none(self):
        """未映射的 tensor pin 返回 None。"""
        tp = TensorPool()
        mp = ModelPool()
        hub = PinHub(node_def=_node_def(0), input_pins={}, tensor_pool=tp, model_pool=mp)
        assert hub.get_tensor(TensorKey.POSITIVE) is None

    def test_get_model_unmapped_raises(self):
        """未映射的 model pin 抛出 KeyError。"""
        tp = TensorPool()
        mp = ModelPool()
        hub = PinHub(node_def=_node_def(0), input_pins={}, tensor_pool=tp, model_pool=mp)
        with pytest.raises(KeyError, match="not connected"):
            hub.get_model(ModelKey.T5TextEncoder)

    def test_read_only_mapped_tensors(self):
        """PinHub 只能读取 input_pins 中声明的上游输出。"""
        tp = TensorPool()
        mp = ModelPool()

        # 在 pool 中直接写入两个 tensor
        key_a = TensorPoolKey(0, TensorKey.POSITIVE)
        key_b = TensorPoolKey(0, TensorKey.NEGATIVE)
        tp.put(key_a, torch.randn(2))
        tp.put(key_b, torch.randn(3))

        # PinHub 只映射 POSITIVE
        hub = PinHub(
            node_def=_node_def(1),
            input_pins={"tensor": {TensorKey.POSITIVE: key_a}},
            tensor_pool=tp,
            model_pool=mp,
        )
        assert hub.get_tensor(TensorKey.POSITIVE) is not None
        # NEGATIVE 未映射，返回 None（即使 pool 中存在）
        assert hub.get_tensor(TensorKey.NEGATIVE) is None

    def test_write_scoped_to_own_node_id(self):
        """put_tensor 写入的 key 自动带上自己的 node_id。"""
        tp = TensorPool()
        mp = ModelPool()

        hub = PinHub(node_def=_node_def(42), input_pins={}, tensor_pool=tp, model_pool=mp)
        hub.put_tensor(TensorKey.VIDEO, torch.randn(1))

        # 验证 pool 中的 key 是 TensorPoolKey(42, VIDEO)
        assert tp.has(TensorPoolKey(42, TensorKey.VIDEO))
        # 其他 node_id 的 key 不存在
        assert not tp.has(TensorPoolKey(0, TensorKey.VIDEO))
        assert not tp.has(TensorPoolKey(1, TensorKey.VIDEO))

    def test_put_and_get_model(self):
        """put_model 写入后，下游 PinHub 通过映射可读回。"""
        tp = TensorPool()
        mp = ModelPool()

        # 上游 node_id=0 写入 T5TextEncoder（有参调用）
        nd_up = _node_def(0, model_key=ModelKey.T5TextEncoder, node_type=IONodeType.LOAD_MODEL)
        upstream = PinHub(node_def=nd_up, input_pins={}, tensor_pool=tp, model_pool=mp)
        model = _StubModel(ModelKey.T5TextEncoder)
        upstream.put_model(model, ModelKey.T5TextEncoder)

        # 下游 node_id=1 映射 T5TextEncoder -> 上游的 ModelPoolKey(0, T5TextEncoder)
        nd_down = _node_def(1, model_key=ModelKey.T5TextEncoder)
        downstream = PinHub(
            node_def=nd_down,
            input_pins={"model": {ModelKey.T5TextEncoder: ModelPoolKey(0, ModelKey.T5TextEncoder)}},
            tensor_pool=tp,
            model_pool=mp,
        )
        result = downstream.get_model(ModelKey.T5TextEncoder)
        assert result is model

    def test_peek_tensor(self):
        """peek_tensor 不消费 tensor，多次调用返回相同数据。"""
        tp = TensorPool()
        mp = ModelPool()

        key = TensorPoolKey(0, TensorKey.LATENTS)
        data = torch.randn(2, 4)
        tp.put(key, data)

        hub = PinHub(
            node_def=_node_def(1),
            input_pins={"tensor": {TensorKey.LATENTS: key}},
            tensor_pool=tp,
            model_pool=mp,
        )
        first = hub.peek_tensor(TensorKey.LATENTS)
        second = hub.peek_tensor(TensorKey.LATENTS)
        assert first is data
        assert second is data

    def test_peek_tensor_unmapped_returns_none(self):
        """未映射的 tensor pin peek 返回 None。"""
        tp = TensorPool()
        mp = ModelPool()
        hub = PinHub(node_def=_node_def(0), input_pins={}, tensor_pool=tp, model_pool=mp)
        assert hub.peek_tensor(TensorKey.POSITIVE) is None

    def test_input_pins_serializable(self):
        """input_pins 可通过 pickle 序列化（模拟 Ray 传输）。"""
        mapping = {
            "tensor": {TensorKey.POSITIVE: TensorPoolKey(1, TensorKey.POSITIVE)},
            "model": {ModelKey.T5TextEncoder: ModelPoolKey(0, ModelKey.T5TextEncoder)},
        }
        restored = pickle.loads(pickle.dumps(mapping))
        assert restored == mapping


class TestPinHubModelAutoResolve:
    """get_model() / put_model() 无参调用 — 自动使用 node_def.model_key。"""

    def test_get_model_no_arg(self):
        """get_model() 无参时自动使用 node_def.model_key。"""
        tp = TensorPool()
        mp = ModelPool()

        # 上游 Loader 写入 model
        nd_up = _node_def(0, model_key=ModelKey.T5TextEncoder, node_type=IONodeType.LOAD_MODEL)
        upstream = PinHub(node_def=nd_up, input_pins={}, tensor_pool=tp, model_pool=mp)
        model = _StubModel(ModelKey.T5TextEncoder)
        upstream.put_model(model)  # 无参 — 自动用 node_def.model_key

        # 下游 Infer 读取 model
        nd_down = _node_def(1, model_key=ModelKey.T5TextEncoder)
        downstream = PinHub(
            node_def=nd_down,
            input_pins={"model": {ModelKey.T5TextEncoder: ModelPoolKey(0, ModelKey.T5TextEncoder)}},
            tensor_pool=tp,
            model_pool=mp,
        )
        result = downstream.get_model()  # 无参 — 自动用 node_def.model_key
        assert result is model

    def test_put_model_no_arg(self):
        """put_model(model) 无参 pin 时自动使用 node_def.model_key。"""
        tp = TensorPool()
        mp = ModelPool()

        nd = _node_def(5, model_key=ModelKey.VAE_WAN2_2, node_type=IONodeType.LOAD_MODEL)
        hub = PinHub(node_def=nd, input_pins={}, tensor_pool=tp, model_pool=mp)
        model = _StubModel(ModelKey.VAE_WAN2_2)
        hub.put_model(model)  # 无参

        # 验证 ModelPool 中有对应 key
        pool_key = ModelPoolKey(5, ModelKey.VAE_WAN2_2)
        assert mp.get_model(pool_key) is model

    def test_get_model_no_arg_no_model_key_raises(self):
        """node_def.model_key 为 None 时，get_model() 无参调用抛出 KeyError。"""
        tp = TensorPool()
        mp = ModelPool()

        nd = _node_def(0, model_key=None)
        hub = PinHub(node_def=nd, input_pins={}, tensor_pool=tp, model_pool=mp)
        with pytest.raises(KeyError, match="not connected"):
            hub.get_model()

    def test_get_model_explicit_overrides_node_def(self):
        """get_model(pin=X) 显式指定时忽略 node_def.model_key。"""
        tp = TensorPool()
        mp = ModelPool()

        # node_def 的 model_key 是 T5，但显式请求 VAE
        nd = _node_def(1, model_key=ModelKey.T5TextEncoder)
        hub = PinHub(
            node_def=nd,
            input_pins={"model": {ModelKey.VAE_WAN2_2: ModelPoolKey(0, ModelKey.VAE_WAN2_2)}},
            tensor_pool=tp,
            model_pool=mp,
        )
        # 先在 pool 中放入 VAE model
        vae_model = _StubModel(ModelKey.VAE_WAN2_2)
        mp.update_model_with_key(ModelPoolKey(0, ModelKey.VAE_WAN2_2), vae_model)

        result = hub.get_model(ModelKey.VAE_WAN2_2)
        assert result is vae_model
