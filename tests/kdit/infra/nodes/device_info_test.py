# Copyright 2026 Tencent

import pytest
import torch

from kdit.nodes.core.device_context import DeviceInfo, NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext


class TestDeviceInfo:
    def test_frozen(self):
        info = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )
        with pytest.raises(AttributeError):
            info.rank_id = 2

    def test_fields_accessible(self):
        info = DeviceInfo(
            device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=4,
        )
        assert info.device == torch.device("cuda:0")
        assert info.offload_device == torch.device("cpu")
        assert info.rank_id == 0
        assert info.world_size == 4

    def test_alias_is_same_class(self):
        assert NodeDeviceContext is DeviceInfo


class TestNodeContextDevice:
    def test_device_default_none(self):
        ctx = NodeContext()
        assert ctx.device is None

    def test_device_assigned(self):
        info = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )
        ctx = NodeContext(device=info)
        assert ctx.device is info
        assert ctx.device.device == torch.device("cpu")
        assert ctx.device.rank_id == 0
