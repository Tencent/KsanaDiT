# Copyright 2026 Tencent

import pytest

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_def import NodeDef, NodeRef
from kdit.nodes.core.node_types import InferNodeType
from kdit.nodes.core.pin_def import PinRef
from kdit.tensor.tensor_key import TensorKey


class TestPinRef:
    def test_tensor_pin_ref(self):
        ref = PinRef(node_id=42, pin=TensorKey.POSITIVE)
        assert ref.node_id == 42
        assert ref.pin is TensorKey.POSITIVE

    def test_model_pin_ref(self):
        ref = PinRef(node_id=7, pin=ModelKey.T5TextEncoder)
        assert ref.node_id == 7
        assert ref.pin is ModelKey.T5TextEncoder

    def test_frozen(self):
        ref = PinRef(node_id=1, pin=TensorKey.POSITIVE)
        with pytest.raises(AttributeError):
            ref.node_id = 2


class TestNodeRef:
    def _make_node_def(self):
        return NodeDef(node_type=InferNodeType.TEXT_ENCODE)

    def test_tensor_key_access(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        ref = node.POSITIVE
        assert isinstance(ref, PinRef)
        assert ref == PinRef(nd.node_id, TensorKey.POSITIVE)

    def test_model_key_access(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        ref = node.T5TextEncoder
        assert isinstance(ref, PinRef)
        assert ref == PinRef(nd.node_id, ModelKey.T5TextEncoder)

    def test_invalid_name_raises(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        with pytest.raises(AttributeError, match="Unknown pin"):
            _ = node.INVALID_NAME

    def test_node_id_property(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        assert node.node_id == nd.node_id

    def test_dir_contains_all_members(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        members = dir(node)
        for tk in TensorKey.__members__:
            assert tk in members
        for mk in ModelKey.__members__:
            assert mk in members

    def test_base_latent_access(self):
        nd = self._make_node_def()
        node = NodeRef(nd)
        ref = node.BASE_LATENT
        assert ref == PinRef(nd.node_id, TensorKey.BASE_LATENT)
