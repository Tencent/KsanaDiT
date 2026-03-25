# Copyright 2026 Tencent

"""Tests for Executor.run_node() — DAG 模式的 Node 执行入口。

这些测试通过 mock 避免真实 GPU 依赖，验证核心逻辑：
- DeviceInfo 注入
- PinHub 构建并作为第一个位置参数传给 node.run()
- context 作为 keyword-only 参数传给 node.run()
- IONode 自动注入 dist_config / shard_fn
- Node 实例缓存
- dispatch policy 处理
"""

from unittest.mock import MagicMock, patch

import torch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool import ModelPool
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.device_context import DeviceInfo
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType, IONodeType, NodeDispatchPolicy
from kdit.nodes.core.pin_hub import PinHub
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor.tensor_key import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey

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
    """测试 run_node() 对 InferNode 的处理。

    新签名：node.run(pins: PinHub, *, context: NodeContext)
    - pins 是第一个位置参数（PinHub 实例）
    - context 是 keyword-only 参数
    """

    def test_device_info_injected_when_none(self):
        """context.device 为 None 时，run_node() 自动注入 executor 的 DeviceInfo。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")
        assert context.device is None

        node_def = NodeDef(node_id=0, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        # 验证 node.run() 被调用，且 context（keyword 参数）中注入了 device
        mock_node.run.assert_called_once()
        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        assert injected_ctx.device is not None
        assert injected_ctx.device == executor.device_ctx

    def test_device_info_not_overwritten_when_present(self):
        """context.device 已有值时，run_node() 不覆盖。"""
        executor = _make_executor()
        existing_device = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=7,
            world_size=8,
        )
        context = NodeContext(prompt="test", device=existing_device)

        node_def = NodeDef(node_id=0, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        # 应保留原始 device，不被覆盖
        assert injected_ctx.device.rank_id == 7
        assert injected_ctx.device.world_size == 8

    def test_infer_node_receives_pin_hub_as_first_arg(self):
        """InferNode 的 run() 第一个位置参数是 PinHub 实例。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=5, node_type=InferNodeType.GENERATE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_args = mock_node.run.call_args
        pins = call_args.args[0]
        assert isinstance(pins, PinHub)

    def test_infer_node_receives_context_as_kwarg(self):
        """InferNode 的 run() 接收 context 作为 keyword-only 参数。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], NodeContext)

    def test_pre_post_sync_called_for_infer_node(self):
        """InferNode 执行时调用 _pre_sync_tensors 和 _post_sync_tensors。"""
        executor = _make_executor()
        executor._pre_sync_tensors = MagicMock()
        executor._post_sync_tensors = MagicMock()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=InferNodeType.VAE_DECODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        executor._pre_sync_tensors.assert_called_once_with(mock_node, NodeDispatchPolicy.ALL_ALL_ALL)
        executor._post_sync_tensors.assert_called_once_with(mock_node, node_def, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_infer_node_call_signature(self):
        """run_node() 调用签名验证：node.run(pins: PinHub, context=NodeContext)。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_args = mock_node.run.call_args
        # 只有 1 个位置参数（PinHub）
        assert len(call_args.args) == 1
        assert isinstance(call_args.args[0], PinHub)
        # keyword 参数只有 context
        assert set(call_args.kwargs.keys()) == {"context"}
        assert isinstance(call_args.kwargs["context"], NodeContext)


class TestExecutorRunLoaderNode:
    """测试 run_node() 对 IONode 的处理。

    新签名：node.run(pins: PinHub, *, context: NodeContext)
    - pins 是第一个位置参数（PinHub 实例）
    - context 是 keyword-only 参数
    - dist_config / shard_fn 注入到 context.metadata
    """

    def test_loader_node_no_sync(self):
        """IONode 不调用 _pre_sync_tensors / _post_sync_tensors。"""
        executor = _make_executor()
        executor._pre_sync_tensors = MagicMock()
        executor._post_sync_tensors = MagicMock()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        # IONode 不需要 tensor sync
        executor._pre_sync_tensors.assert_not_called()
        executor._post_sync_tensors.assert_not_called()

    def test_loader_node_receives_pin_hub_as_first_arg(self):
        """IONode 的 run() 第一个位置参数是 PinHub 实例。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_args = mock_node.run.call_args
        pins = call_args.args[0]
        assert isinstance(pins, PinHub)

    def test_loader_node_receives_context_as_kwarg(self):
        """IONode 的 run() 接收 context 作为 keyword-only 参数。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], NodeContext)

    def test_loader_node_dist_config_injected_to_metadata(self):
        """dist_config 和 shard_fn 被注入到 context.metadata。"""
        executor = _make_executor()
        context = NodeContext(prompt="test", metadata={})

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

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

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.VAE_WAN2_1)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        assert injected_ctx.metadata["model_path"] == "/tmp/vae.safetensors"
        assert injected_ctx.metadata["extra_flag"] is True

    def test_loader_node_metadata_does_not_override_dist_config(self):
        """metadata 中已有的 dist_config 不被覆盖（setdefault 语义）。"""
        executor = _make_executor()
        fake_dist = MagicMock()
        context = NodeContext(prompt="test", metadata={"dist_config": fake_dist})

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        call_kwargs = mock_node.run.call_args.kwargs
        injected_ctx = call_kwargs["context"]
        # metadata 中已有 dist_config，setdefault 不覆盖，保留 metadata 的值
        assert injected_ctx.metadata["dist_config"] is fake_dist

    def test_loader_node_call_signature(self):
        """run_node() 调用签名验证：node.run(pins: PinHub, context=NodeContext)。"""
        executor = _make_executor()
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

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
        """IONode 通过 LoaderNodeFactory.create() 创建。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        with patch("kdit.nodes.core.node_factory.LoaderNodeFactory.create", return_value=mock_node) as mock_create:
            result = executor._get_or_create_node(node_def)

        mock_create.assert_called_once_with(ModelKey.T5TextEncoder)
        assert result is mock_node

    def test_infer_node_created_via_factory(self):
        """InferNode 通过 InferNodeFactory.create() 创建。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=1, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        with patch("kdit.nodes.core.node_factory.InferNodeFactory.create", return_value=mock_node) as mock_create:
            result = executor._get_or_create_node(node_def)

        mock_create.assert_called_once_with(InferNodeType.TEXT_ENCODE, ModelKey.T5TextEncoder)
        assert result is mock_node

    def test_node_cached_by_node_id(self):
        """同一 node_id 只创建一次，后续调用返回缓存实例。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=42, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

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

        node_def = NodeDef(node_id=0, node_type=InferNodeType.VAE_ENCODE_SPATIAL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

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

        node_def = NodeDef(node_id=0, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
        mock_node.output_defs = []

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        mock_node.run.assert_called_once()

    def test_loader_r0_only_skips_non_rank0(self):
        """IONode 的 R0_R0_BCAST policy 下，非 rank0 不执行。"""
        executor = _make_executor()
        executor.device_ctx = DeviceInfo(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=1,
            world_size=2,
        )
        context = NodeContext(prompt="test")

        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins={}, context=context)

        mock_node.run.assert_not_called()


class TestBuildOutputPins:
    """测试 _build_output_pins() 从 Node 的 output_defs 构建 output_pins。"""

    def test_output_pins_built_from_output_defs(self):
        """InferNode 的 output_pins 包含 output_defs 中的 TensorKey。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=5, node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)

        mock_node = MagicMock()
        mock_node.output_defs = [TensorKey.LATENTS]
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            output_pins = executor.run_node(node_def, input_pins={}, context=NodeContext(prompt="test"))

        assert TensorKey.LATENTS in output_pins
        assert output_pins[TensorKey.LATENTS] == TensorPoolKey(5, TensorKey.LATENTS)

    def test_output_pins_includes_model_for_loader(self):
        """IONode（Loader）的 output_pins 包含 model 映射。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=2, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.VAE_WAN2_2)

        mock_node = MagicMock()
        mock_node.output_defs = []
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            output_pins = executor.run_node(node_def, input_pins={}, context=NodeContext(metadata={}))

        assert ModelKey.VAE_WAN2_2 in output_pins
        assert output_pins[ModelKey.VAE_WAN2_2] == ModelPoolKey(2, ModelKey.VAE_WAN2_2)

    def test_output_pins_empty_for_no_outputs(self):
        """没有 output_defs 且 model_key 为 None 的 InferNode 返回空 dict。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=10, node_type=InferNodeType.SAVE_VIDEO)

        mock_node = MagicMock()
        mock_node.output_defs = []
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            output_pins = executor.run_node(node_def, input_pins={}, context=NodeContext(prompt="test"))

        assert output_pins == {}

    def test_output_pins_multiple_tensor_keys(self):
        """多个 TensorKey 的 output_defs 全部映射到 output_pins。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=3, node_type=InferNodeType.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.output_defs = [TensorKey.POSITIVE, TensorKey.NEGATIVE]
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            output_pins = executor.run_node(node_def, input_pins={}, context=NodeContext(prompt="test"))

        assert output_pins[TensorKey.POSITIVE] == TensorPoolKey(3, TensorKey.POSITIVE)
        assert output_pins[TensorKey.NEGATIVE] == TensorPoolKey(3, TensorKey.NEGATIVE)


class TestAutoConsumeInputTensors:
    """测试 run_node() 自动消费输入 tensor 引用。"""

    def test_consume_called_for_input_tensors(self):
        """run_node() 执行后自动 consume 输入 tensor 的引用。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=5, node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)

        # 注册引用计数
        input_pool_key = TensorPoolKey(3, TensorKey.POSITIVE)
        executor.tensor_pool.put(input_pool_key, MagicMock())
        executor.tensor_pool.register(input_pool_key, ref_count=2)

        mock_node = MagicMock()
        mock_node.output_defs = [TensorKey.LATENTS]
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        input_pins = {TensorKey.POSITIVE: input_pool_key}

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins=input_pins, context=NodeContext(prompt="test"))

        # ref_count 从 2 降到 1，tensor 仍存在
        assert executor.tensor_pool.get(input_pool_key) is not None

    def test_consume_releases_when_ref_count_zero(self):
        """引用计数归零时，tensor 自动释放。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=5, node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)

        input_pool_key = TensorPoolKey(3, TensorKey.POSITIVE)
        executor.tensor_pool.put(input_pool_key, MagicMock())
        executor.tensor_pool.register(input_pool_key, ref_count=1)

        mock_node = MagicMock()
        mock_node.output_defs = []
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        input_pins = {TensorKey.POSITIVE: input_pool_key}

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins=input_pins, context=NodeContext(prompt="test"))

        # ref_count 从 1 降到 0，tensor 被释放
        assert executor.tensor_pool.get(input_pool_key) is None

    def test_loader_does_not_consume_inputs(self):
        """run_node() 不自动消费输入 tensor（Loader 没有 tensor 输入）。"""
        executor = _make_executor()
        node_def = NodeDef(node_id=0, node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)

        mock_node = MagicMock()
        mock_node.output_defs = []
        mock_node.dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

        # 即使 input_pins 中有 tensor，Loader 也不消费
        input_pool_key = TensorPoolKey(99, TensorKey.POSITIVE)
        executor.tensor_pool.put(input_pool_key, MagicMock())
        executor.tensor_pool.register(input_pool_key, ref_count=1)

        input_pins = {TensorKey.POSITIVE: input_pool_key}

        with patch("kdit.executor.executor.Executor._get_or_create_node", return_value=mock_node):
            executor.run_node(node_def, input_pins=input_pins, context=NodeContext(metadata={}))

        # Loader 不消费，tensor 仍存在
        assert executor.tensor_pool.get(input_pool_key) is not None


class TestRegisterRefCount:
    """测试 register_ref_count() 透传到 tensor_pool.register()。"""

    def test_register_ref_count(self):
        """register_ref_count() 正确注册引用计数。"""
        executor = _make_executor()
        pool_key = TensorPoolKey(5, TensorKey.LATENTS)
        executor.tensor_pool.put(pool_key, MagicMock())

        executor.register_ref_count(pool_key, ref_count=3)

        # consume 3 次后 tensor 被释放
        executor.tensor_pool.consume(pool_key)
        assert executor.tensor_pool.get(pool_key) is not None
        executor.tensor_pool.consume(pool_key)
        assert executor.tensor_pool.get(pool_key) is not None
        executor.tensor_pool.consume(pool_key)
        assert executor.tensor_pool.get(pool_key) is None
