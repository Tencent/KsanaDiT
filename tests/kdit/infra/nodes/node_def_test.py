# Copyright 2026 Tencent

"""NodeDef 自动 ID 分配 + NodeRef 测试。"""

import pytest

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.node_types import InferNodeType, IONodeType


class TestNodeDefAutoId:
    """node_id 自动分配测试。"""

    def test_auto_id_unique(self):
        """每个 NodeDef 的 node_id 不重复。"""
        a = NodeDef(node_type=InferNodeType.TEXT_ENCODE)
        b = NodeDef(node_type=InferNodeType.GENERATE)
        c = NodeDef(node_type=InferNodeType.VAE_DECODE)
        ids = {a.node_id, b.node_id, c.node_id}
        assert len(ids) == 3

    def test_auto_id_monotonic(self):
        """node_id 单调递增。"""
        a = NodeDef(node_type=InferNodeType.TEXT_ENCODE)
        b = NodeDef(node_type=InferNodeType.GENERATE)
        assert b.node_id > a.node_id

    def test_no_explicit_id(self):
        """不能通过构造参数指定 node_id。"""
        with pytest.raises(TypeError):
            NodeDef(node_id=42, node_type=InferNodeType.TEXT_ENCODE)

    def test_frozen(self):
        """NodeDef 是 frozen dataclass，不可修改字段。"""
        nd = NodeDef(node_type=InferNodeType.TEXT_ENCODE)
        with pytest.raises(AttributeError):
            nd.node_type = InferNodeType.GENERATE

    def test_is_io_loader(self):
        """IONodeType 的 is_io / is_loader 返回 True。"""
        nd = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        assert nd.is_io is True
        assert nd.is_loader is True

    def test_is_io_infer(self):
        """InferNodeType 的 is_io / is_loader 返回 False。"""
        nd = NodeDef(node_type=InferNodeType.TEXT_ENCODE)
        assert nd.is_io is False
        assert nd.is_loader is False

    def test_condition_field(self):
        """condition 字段正确设置。"""
        nd = NodeDef(node_type=InferNodeType.GENERATE, condition="should_run")
        assert nd.condition == "should_run"

    def test_model_key_default_none(self):
        """model_key 默认为 None。"""
        nd = NodeDef(node_type=InferNodeType.SAVE_VIDEO)
        assert nd.model_key is None
