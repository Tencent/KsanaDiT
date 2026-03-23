# Copyright 2026 Tencent

import pytest

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.tensor.tensor_key import TensorKey
from kdit.tensor.tensor_pool_key import TensorPoolKey


class TestModelPoolKey:
    def test_hash_and_eq_same_fields(self):
        a = ModelPoolKey(node_id=0, pin=ModelKey.T5TextEncoder)
        b = ModelPoolKey(node_id=0, pin=ModelKey.T5TextEncoder)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_node_id_not_equal(self):
        a = ModelPoolKey(node_id=0, pin=ModelKey.T5TextEncoder)
        b = ModelPoolKey(node_id=1, pin=ModelKey.T5TextEncoder)
        assert a != b

    def test_different_pin_not_equal(self):
        a = ModelPoolKey(node_id=0, pin=ModelKey.T5TextEncoder)
        b = ModelPoolKey(node_id=0, pin=ModelKey.VAE_WAN2_1)
        assert a != b

    def test_as_dict_key(self):
        key = ModelPoolKey(node_id=3, pin=ModelKey.VAE_WAN2_2)
        d = {key: "hello"}
        assert d[ModelPoolKey(node_id=3, pin=ModelKey.VAE_WAN2_2)] == "hello"

    def test_uid_format(self):
        key = ModelPoolKey(node_id=5, pin=ModelKey.T5TextEncoder)
        assert key.uid == "model:5:T5TextEncoder"

    def test_frozen(self):
        key = ModelPoolKey(node_id=0, pin=ModelKey.T5TextEncoder)
        with pytest.raises(AttributeError):
            key.node_id = 1


class TestTensorPoolKey:
    def test_hash_and_eq_same_fields(self):
        a = TensorPoolKey(node_id=0, pin=TensorKey.POSITIVE)
        b = TensorPoolKey(node_id=0, pin=TensorKey.POSITIVE)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_node_id_not_equal(self):
        a = TensorPoolKey(node_id=0, pin=TensorKey.POSITIVE)
        b = TensorPoolKey(node_id=1, pin=TensorKey.POSITIVE)
        assert a != b

    def test_different_pin_not_equal(self):
        a = TensorPoolKey(node_id=0, pin=TensorKey.POSITIVE)
        b = TensorPoolKey(node_id=0, pin=TensorKey.NEGATIVE)
        assert a != b

    def test_as_dict_key(self):
        key = TensorPoolKey(node_id=2, pin=TensorKey.LATENTS)
        d = {key: 42}
        assert d[TensorPoolKey(node_id=2, pin=TensorKey.LATENTS)] == 42

    def test_uid_format(self):
        key = TensorPoolKey(node_id=7, pin=TensorKey.VIDEO)
        assert key.uid == "tensor:7:VIDEO"

    def test_frozen(self):
        key = TensorPoolKey(node_id=0, pin=TensorKey.POSITIVE)
        with pytest.raises(AttributeError):
            key.node_id = 1
