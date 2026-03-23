# Copyright 2026 Tencent

import pytest

from kdit.models.model_key import ModelKey
from kdit.pipelines.pin_ref import NodeRef, PinRef
from kdit.tensor.tensor_key import TensorKey


class TestPinRef:
    def test_tensor_pin_ref(self):
        ref = PinRef(node_id=0, pin=TensorKey.POSITIVE)
        assert ref.node_id == 0
        assert ref.pin is TensorKey.POSITIVE

    def test_model_pin_ref(self):
        ref = PinRef(node_id=1, pin=ModelKey.T5TextEncoder)
        assert ref.node_id == 1
        assert ref.pin is ModelKey.T5TextEncoder

    def test_frozen(self):
        ref = PinRef(node_id=0, pin=TensorKey.POSITIVE)
        with pytest.raises(AttributeError):
            ref.node_id = 1


class TestNodeRef:
    def test_tensor_key_access(self):
        node = NodeRef(0)
        ref = node.POSITIVE
        assert isinstance(ref, PinRef)
        assert ref == PinRef(0, TensorKey.POSITIVE)

    def test_model_key_access(self):
        node = NodeRef(0)
        ref = node.T5TextEncoder
        assert isinstance(ref, PinRef)
        assert ref == PinRef(0, ModelKey.T5TextEncoder)

    def test_invalid_name_raises(self):
        node = NodeRef(0)
        with pytest.raises(AttributeError, match="Unknown pin"):
            _ = node.INVALID_NAME

    def test_node_id_property(self):
        node = NodeRef(42)
        assert node.node_id == 42

    def test_dir_contains_all_members(self):
        node = NodeRef(0)
        members = dir(node)
        for tk in TensorKey.__members__:
            assert tk in members
        for mk in ModelKey.__members__:
            assert mk in members

    def test_base_latent_access(self):
        node = NodeRef(3)
        ref = node.BASE_LATENT
        assert ref == PinRef(3, TensorKey.BASE_LATENT)
