# Copyright 2026 Tencent

"""Tests for Executor.run_loader_node() / run_infer_node() — DAG 模式的 Node 执行入口。

这些测试通过 mock 避免真实 GPU 依赖，验证核心逻辑：
- DeviceInfo 注入
- PinHub 构建并作为第一个位置参数传给 node.run()
- context 作为 keyword-only 参数传给 node.run()
- LoaderNode 自动注入 dist_config / shard_fn
- Node 实例缓存
- dispatch policy 处理
"""

from unittest.mock import MagicMock, patch

import torch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool import ModelPool
from kdit.nodes.core.device_context import DeviceInfo
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType, NodeDispatchPolicy
from kdit.nodes.core.pin_hub import PinHub
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor.tensor_pool import TensorPool

# ── 辅助：构建一个最小化的 Executor（绕过 __init__ 中的 CUDA 调用）──


def _make_executor():
    """构建一个最小化的 Executor 实例，绕过 __init__ 中的 CUDA 调用。"""
    from kdit.config import DistributedConfig
    from kdit.executor.executor import Executor

    # 绕过 ABC 限制和 __init__ 中的 CUDA 调用
    executor = object.__new__(Executor)
    executor.device_id = 0
    executor.rank_id = 0
    executor.world_size = 1
    executor.device = torch.device("cpu")
    executor.offload_device = torch.device("cpu")
    executor.model_pool = ModelPool()
    executor.shard_fn = None
    executor.dist_config = DistributedConfig(num_gpus=1, use_sp=False, dit_fsdp=False, ulysses_size=1)
    executor.tensor_pool = TensorPool()
    executor.dist_group = MagicMock()
    executor.device_ctx = DeviceInfo(
        device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
        rank_id=0,
        world_size=1,
    )
    executor._node_cache = {}
    return executor


# ── 测试类 ──


class TestExecutorRunInferNode:
    """测试 run_infer_node() 对 InferNode 的处理。

    新签名：node.run(pins: PinHub, *, context: NodeContext)
    - pins 是第一个位置参数（PinHub 实例）
    - context 是 keyword-only 参数
    """

    def test_device_info_injected_when_none(self):
        """context.device 为 None 时，run_infer_node() 自动注入 executor 的 DeviceInfo。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")
        assert context.device is None

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        # 验证 node.run() 被调用，且 context（keyword 参数）中注入了 device
        mock_node.run.assert_called_once()
        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        assert injected_ctx.device is not None
        assert injected_ctx.device == executor.device_ctx

    def test_device_info_not_overwritten_when_present(self):
        """context.device 已有值时，run_infer_node() 不覆盖。"""
        executor = _make_executor()
        existing_device = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=7,
            world_size=8,
        )
        context = NodeContext(prompt="test", device=existing_device)

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        # 应保留原始 device，不被覆盖
        assert injected_ctx.device.rank_id == 7
        assert injected_ctx.device.world_size == 8

    def test_infer_node_receives_pin_hub_as_first_arg(self):
        """InferNode 的 run() 第一个位置参数是 PinHub 实例。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=5, is_loader=False, node_type=InferNodeType.GENERATE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_args = mock_node.run.call_args
        pins = call_args.args[0]
        assert isinstance(pins, PinHub)

    def test_infer_node_receives_context_as_kwarg(self):
        """InferNode 的 run() 接收 context 作为 keyword-only 参数。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], NodeContext)

    def test_pre_post_sync_called_for_infer_node(self):
        """InferNode 执行时调用 _pre_sync_tensors 和 _post_sync_tensors。"""
        executor = _make_executor()
        executor._pre_sync_tensors = MagicMock()
        executor._post_sync_tensors = MagicMock()
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.VAE_DECODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        executor._pre_sync_tensors.assert_called_once_with(mock_node, NodeDispatchPolicy.ALL_ALL_ALL)
        executor._post_sync_tensors.assert_called_once_with(mock_node, node_def, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_infer_node_call_signature(self):
        """run_infer_node() 调用签名验证：node.run(pins: PinHub, context=NodeContext)。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_args = mock_node.run.call_args
        # 只有 1 个位置参数（PinHub）
        assert len(call_args.args) == 1
        assert isinstance(call_args.args[0], PinHub)
        # keyword 参数只有 context
        assert set(call_args.kwargs.keys()) == {"context"}
        assert isinstance(call_args.kwargs["context"], NodeContext)


class TestExecutorRunLoaderNode:
    """测试 run_loader_node() 对 LoaderNode 的处理。

    新签名：node.run(pins: PinHub, *, context: NodeContext)
    - pins 是第一个位置参数（PinHub 实例）
    - context 是 keyword-only 参数
    - dist_config / shard_fn 注入到 context.metadata
    """

    def test_loader_node_no_sync(self):
        """LoaderNode 不调用 _pre_sync_tensors / _post_sync_tensors。"""
        executor = _make_executor()
        executor._pre_sync_tensors = MagicMock()
        executor._post_sync_tensors = MagicMock()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        # LoaderNode 不需要 tensor sync
        executor._pre_sync_tensors.assert_not_called()
        executor._post_sync_tensors.assert_not_called()

    def test_loader_node_receives_pin_hub_as_first_arg(self):
        """LoaderNode 的 run() 第一个位置参数是 PinHub 实例。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_args = mock_node.run.call_args
        pins = call_args.args[0]
        assert isinstance(pins, PinHub)

    def test_loader_node_receives_context_as_kwarg(self):
        """LoaderNode 的 run() 接收 context 作为 keyword-only 参数。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], NodeContext)

    def test_loader_node_dist_config_injected_to_metadata(self):
        """dist_config 和 shard_fn 被注入到 context.metadata。"""
        executor = _make_executor()
        context = NodeContext(prompt="test", metadata={})

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        assert "dist_config" in injected_ctx.metadata
        assert "shard_fn" in injected_ctx.metadata
        assert injected_ctx.metadata["dist_config"] == executor.dist_config
        assert injected_ctx.metadata["shard_fn"] == executor.shard_fn

    def test_loader_node_metadata_preserved(self):
        """context.metadata 中已有的键值对被保留。"""
        executor = _make_executor()
        context = NodeContext(prompt="test", metadata={"model_path": "/tmp/vae.safetensors", "extra_flag": True})

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.VAE_WAN2_1)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        assert injected_ctx.metadata["model_path"] == "/tmp/vae.safetensors"
        assert injected_ctx.metadata["extra_flag"] is True

    def test_loader_node_metadata_does_not_override_dist_config(self):
        """metadata 中已有的 dist_config 不被覆盖（setdefault 语义）。"""
        executor = _make_executor()
        fake_dist = MagicMock()
        context = NodeContext(prompt="test", metadata={"dist_config": fake_dist})

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        # metadata 中已有 dist_config，setdefault 不覆盖，保留 metadata 的值
        assert injected_ctx.metadata["dist_config"] is fake_dist

    def test_loader_node_call_signature(self):
        """run_loader_node() 调用签名验证：node.run(pins: PinHub, context=NodeContext)。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        call_args = mock_node.run.call_args
        # 只有 1 个位置参数（PinHub）
        assert len(call_args.args) == 1
        assert isinstance(call_args.args[0], PinHub)
        # keyword 参数只有 context
        assert set(call_args.kwargs.keys()) == {"context"}
        assert isinstance(call_args.kwargs["context"], NodeContext)


class TestGetOrCreateNode:
    """测试 _get_or_create_node() 的缓存行为。"""

    def test_loader_node_created_via_factory(self):
        """LoaderNode 通过 LoaderNodeFactory.create() 创建。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        with patch("kdit.nodes.core.node_factory.LoaderNodeFactory.create", return_value=mock_node) as mock_create:
            result = executor._get_or_create_node(node_def)

        mock_create.assert_called_once_with(ModelKey.T5TextEncoder)
        assert result is mock_node

    def test_infer_node_created_via_factory(self):
        """InferNode 通过 InferNodeFactory.create() 创建。"""
        executor = _make_executor()
        node_def = NodeDef(
            node_id=1, is_loader=False, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        with patch("kdit.nodes.core.node_factory.InferNodeFactory.create", return_value=mock_node) as mock_create:
            result = executor._get_or_create_node(node_def)

        mock_create.assert_called_once_with(InferNodeType.TEXT_ENCODE, ModelKey.T5TextEncoder)
        assert result is mock_node

    def test_node_cached_by_node_id(self):
        """同一 node_id 只创建一次，后续调用返回缓存实例。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=42, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        with patch("kdit.nodes.core.node_factory.LoaderNodeFactory.create", return_value=mock_node) as mock_create:
            first = executor._get_or_create_node(node_def)
            second = executor._get_or_create_node(node_def)

        # Factory 只调用一次
        mock_create.assert_called_once()
        assert first is second
        assert first is mock_node


class TestDispatchPolicySkip:
    """测试 dispatch policy 导致的跳过执行。"""

    def test_r0_r0_bcast_skips_non_rank0_infer(self):
        """R0_R0_BCAST policy 下，非 rank0 的 InferNode 不执行 run()。"""
        executor = _make_executor()
        # 模拟 rank 1
        executor.device_ctx = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=1,
            world_size=2,
        )
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.VAE_ENCODE_SPATIAL, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        # run() 不应被调用（非 rank0）
        mock_node.run.assert_not_called()

    def test_all_all_all_executes_on_all_ranks(self):
        """ALL_ALL_ALL policy 下，所有 rank 都执行 run()。"""
        executor = _make_executor()
        # 模拟 rank 3
        executor.device_ctx = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=3,
            world_size=4,
        )
        context = NodeContext(prompt="test")

        node_def = NodeDef(
            node_id=0, is_loader=False, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder
        )

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_tensor_pins = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_infer_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        mock_node.run.assert_called_once()

    def test_loader_r0_only_skips_non_rank0(self):
        """LoaderNode 的 R0_R0_BCAST policy 下，非 rank0 不执行。"""
        executor = _make_executor()
        executor.device_ctx = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=1,
            world_size=2,
        )
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_loader_node(node_def, pins_mapping={"model": {}, "tensor": {}}, context=context)

        mock_node.run.assert_not_called()
