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

"""Node 层单元测试 — NodeDispatchPolicy, InferNodeType, NodeDeviceContext, NodeContext, NodeFactory。"""

import pytest
import torch

from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_factory import InferNodeFactory, LoaderNodeFactory
from kdit.nodes.core.node_types import InferNodeType, NodeDispatchPolicy

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
            "VAE_COMPUTE_SHAPE",
            "VAE_ENCODE_SPATIAL",
            "VAE_ENCODE_IMAGES",
            "VAE_DECODE",
            "GENERATE",
            "SAVE_VIDEO",
            "SAVE_IMAGE",
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
