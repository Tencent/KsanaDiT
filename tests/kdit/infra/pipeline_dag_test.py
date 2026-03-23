# Copyright 2026 Tencent
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

"""Pipeline DAG 遍历逻辑 单元测试。

测试 Pipeline 的 load_models() 和 generate() 的行为：
- 按拓扑序遍历 Loader / Infer 节点
- load_models() 调用 engine.run_loader_node()
- generate() 调用 engine.run_infer_node()
- pins_mapping 正确传递
- 条件跳过（check_condition）的行为
"""

from unittest.mock import MagicMock, patch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.pipeline import Pipeline, _node_def_display_name, _phase_display_name
from kdit.pipelines.pipeline_def import Edge, NodeDef, PipelineDef
from kdit.pipelines.pipeline_key import PipelineKey
from kdit.pipelines.pipeline_phase import InferTask
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool_key import TensorPoolKey

# ── 辅助：构建 mock 对象 ──────────────────────────────────────────────────


def _make_mock_engine():
    """构建一个 mock Engine，模拟所有 Pipeline 需要的方法。"""
    engine = MagicMock()
    engine.tensor_scope = MagicMock()
    # tensor_scope 作为 context manager
    engine.tensor_scope.return_value.__enter__ = MagicMock(return_value=None)
    engine.tensor_scope.return_value.__exit__ = MagicMock(return_value=False)
    # get_tensor 返回 mock TensorValue
    mock_tv = MagicMock()
    mock_tv.data = "fake_output"
    engine.get_tensor.return_value = mock_tv
    return engine


def _make_mock_ctx_builder():
    """构建一个 mock ContextBuilder。"""
    ctx_builder = MagicMock()
    ctx_builder.build_loader_kwargs.return_value = {"model_path": "/fake/path"}
    ctx_builder.build_context.return_value = NodeContext(prompt="test")
    ctx_builder.prepare_tensors.return_value = None
    ctx_builder.check_condition.return_value = True
    ctx_builder.post_process.side_effect = lambda output, inputs: output
    ctx_builder.prepare_generate_inputs.return_value = None
    ctx_builder.resolve_model_paths.return_value = ("/fake", "/fake/text", "/fake/vae")
    ctx_builder.resolve_lora_config.return_value = None
    return ctx_builder


def _make_dag_pipeline_def(*, with_condition=False):
    """构建一个 DAG 模式的 PipelineDef。

    DAG 结构:
        loader_t5(0) ──model──> text_enc(3)
        loader_dit(1) ──model──> gen(4)
        loader_vae(2) ──model──> vae_dec(5)
        text_enc(3) ──POSITIVE──> gen(4)
        text_enc(3) ──NEGATIVE──> gen(4)
        gen(4) ──LATENTS──> vae_dec(5)
        vae_dec(5) ──VIDEO──> save(6)
    """
    nodes = (
        NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder),
        NodeDef(node_id=1, is_loader=True, model_key=ModelKey.Wan2_2_T2V_14B),
        NodeDef(node_id=2, is_loader=True, model_key=ModelKey.VAE_WAN2_2),
        NodeDef(node_id=3, is_loader=False, node_type=NT.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder),
        NodeDef(
            node_id=4,
            is_loader=False,
            node_type=NT.GENERATE,
            model_key=ModelKey.Wan2_2_T2V_14B,
            condition="should_generate" if with_condition else None,
        ),
        NodeDef(node_id=5, is_loader=False, node_type=NT.VAE_DECODE, model_key=ModelKey.VAE_WAN2_2),
        NodeDef(node_id=6, is_loader=False, node_type=NT.SAVE_VIDEO),
    )
    edges = (
        Edge(0, ModelKey.T5TextEncoder, 3, ModelKey.T5TextEncoder, "model"),
        Edge(1, ModelKey.Wan2_2_T2V_14B, 4, ModelKey.Wan2_2_T2V_14B, "model"),
        Edge(2, ModelKey.VAE_WAN2_2, 5, ModelKey.VAE_WAN2_2, "model"),
        Edge(3, TensorKey.POSITIVE, 4, TensorKey.POSITIVE, "tensor"),
        Edge(3, TensorKey.NEGATIVE, 4, TensorKey.NEGATIVE, "tensor"),
        Edge(4, TensorKey.LATENTS, 5, TensorKey.LATENTS, "tensor"),
        Edge(5, TensorKey.VIDEO, 6, TensorKey.VIDEO, "tensor"),
    )
    mock_ctx_cls = MagicMock
    return PipelineDef(
        pipeline_key=PipelineKey.Wan2_2_T2V_14B,
        nodes=nodes,
        edges=edges,
        keep_tensors=(TensorKey.VIDEO,),
        context_builder_cls=mock_ctx_cls,
    )


def _make_pipeline(pipeline_def, engine=None, ctx_builder=None):
    """构建 Pipeline 实例，注入 mock 依赖。"""
    engine = engine or _make_mock_engine()
    pipeline = Pipeline(pipeline_def, engine)
    if ctx_builder:
        pipeline._ctx_builder = ctx_builder
    return pipeline


def _make_default_settings():
    """构建一个合法的 _default_settings mock，使 _prepare_configs 不报错。"""
    settings = MagicMock()
    # sample_config 子属性需要返回 None 以避免被合并到 SampleConfig 中
    settings.sample_config.steps = None
    settings.sample_config.shift = None
    settings.sample_config.denoise = None
    settings.sample_config.cfg_scale = None
    settings.sample_config.solver = None
    # runtime_config 子属性
    settings.runtime_config.target_size = None
    settings.runtime_config.frame_num = None
    # cache
    settings.cache = None
    return settings


# ── 显示名称测试 ──────────────────────────────────────────────────────────


class TestNodeDefDisplayName:
    """_node_def_display_name() 和 _phase_display_name() 对 NodeDef 的处理。"""

    def test_loader_node_def(self):
        """Loader NodeDef 显示为 LOAD(model_name)。"""
        nd = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        assert _node_def_display_name(nd) == "LOAD(T5TextEncoder)"

    def test_loader_node_def_no_model_key(self):
        """Loader NodeDef 无 model_key 时显示 LOAD(UNKNOWN)。"""
        nd = NodeDef(node_id=0, is_loader=True)
        assert _node_def_display_name(nd) == "LOAD(UNKNOWN)"

    def test_infer_node_def_with_model(self):
        """Infer NodeDef 有 model_key 时显示 TYPE(model_name)。"""
        nd = NodeDef(node_id=1, is_loader=False, node_type=NT.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)
        assert _node_def_display_name(nd) == "GENERATE(Wan2_2_T2V_14B)"

    def test_infer_node_def_no_model(self):
        """Infer NodeDef 无 model_key 时只显示 TYPE。"""
        nd = NodeDef(node_id=2, is_loader=False, node_type=NT.SAVE_VIDEO)
        assert _node_def_display_name(nd) == "SAVE_VIDEO"

    def test_infer_node_def_no_type_no_model(self):
        """Infer NodeDef 无 node_type 时显示 node_id。"""
        nd = NodeDef(node_id=99, is_loader=False)
        assert _node_def_display_name(nd) == "node_99"

    def test_phase_display_name_dispatches_to_node_def(self):
        """_phase_display_name() 对 NodeDef 类型正确分发。"""
        nd = NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder)
        assert _phase_display_name(nd) == "LOAD(T5TextEncoder)"


# ── load_models() 测试 ────────────────────────────────────────────────────


class TestLoadModels:
    """Pipeline.load_models() 的行为。"""

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_calls_run_loader_node_for_loaders_only(self, mock_settings):
        """load_models() 只对 is_loader=True 的节点调用 engine.run_loader_node()。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)

        pipeline.load_models("/fake/path")

        # 应该调用 3 次 run_loader_node（3 个 loader）
        assert engine.run_loader_node.call_count == 3

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_topo_order(self, mock_settings):
        """loader 按拓扑序执行。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)

        pipeline.load_models("/fake/path")

        # 提取所有 run_loader_node 调用的 node_def
        call_node_defs = [c.args[0] for c in engine.run_loader_node.call_args_list]
        # 所有都是 loader
        assert all(nd.is_loader for nd in call_node_defs)
        # node_id 应该是 0, 1, 2（按拓扑序）
        assert [nd.node_id for nd in call_node_defs] == [0, 1, 2]

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_pins_mapping_passed(self, mock_settings):
        """pins_mapping 正确传递给 engine.run_loader_node()。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)

        pipeline.load_models("/fake/path")

        # Loader 节点没有入边，pins_mapping 应该是空的
        for c in engine.run_loader_node.call_args_list:
            pins_mapping = c.args[1]
            assert pins_mapping == {"tensor": {}, "model": {}}

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_context_has_loader_kwargs(self, mock_settings):
        """context.metadata 包含 build_loader_kwargs() 的结果。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        ctx_builder.build_loader_kwargs.return_value = {"model_path": "/test/model"}
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)

        pipeline.load_models("/fake/path")

        # 每次 run_loader_node 的 context 都应该有 metadata
        for c in engine.run_loader_node.call_args_list:
            context = c.args[2]
            assert isinstance(context, NodeContext)
            assert context.metadata == {"model_path": "/test/model"}

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_build_loader_kwargs_called_per_loader(self, mock_settings):
        """build_loader_kwargs() 为每个 loader 调用一次。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)

        pipeline.load_models("/fake/path")

        assert ctx_builder.build_loader_kwargs.call_count == 3
        # 验证 model_key 参数
        called_keys = [c.args[0] for c in ctx_builder.build_loader_kwargs.call_args_list]
        assert called_keys == [ModelKey.T5TextEncoder, ModelKey.Wan2_2_T2V_14B, ModelKey.VAE_WAN2_2]


# ── generate() 测试 ───────────────────────────────────────────────────────


class TestGenerate:
    """Pipeline.generate() 的行为。"""

    def _setup_pipeline_for_generate(self, *, with_condition=False):
        """构建一个可以调用 generate() 的 Pipeline。"""
        pipeline_def = _make_dag_pipeline_def(with_condition=with_condition)
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)
        pipeline._default_settings = _make_default_settings()
        return pipeline, engine, ctx_builder

    def test_generate_calls_run_infer_node_for_infer_only(self):
        """generate() 只对 is_loader=False 的节点调用 engine.run_infer_node()。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        # 应该调用 4 次 run_infer_node（4 个 infer 节点: text_enc, gen, vae_dec, save）
        assert engine.run_infer_node.call_count == 4

    def test_generate_topo_order(self):
        """infer 节点按拓扑序执行。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        call_node_defs = [c.args[0] for c in engine.run_infer_node.call_args_list]
        # 所有都是 infer 节点
        assert all(not nd.is_loader for nd in call_node_defs)
        # 拓扑序: text_enc(3) → gen(4) → vae_dec(5) → save(6)
        assert [nd.node_id for nd in call_node_defs] == [3, 4, 5, 6]

    def test_generate_pins_mapping_correct(self):
        """infer 节点的 pins_mapping 正确。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        calls = engine.run_infer_node.call_args_list

        # text_enc(3): 入边 = loader_t5(0) → model
        pm_text = calls[0].args[1]
        assert pm_text["model"] == {ModelKey.T5TextEncoder: ModelPoolKey(0, ModelKey.T5TextEncoder)}
        assert pm_text["tensor"] == {}

        # gen(4): 入边 = loader_dit(1) → model, text_enc(3) → POSITIVE, text_enc(3) → NEGATIVE
        pm_gen = calls[1].args[1]
        assert pm_gen["model"] == {ModelKey.Wan2_2_T2V_14B: ModelPoolKey(1, ModelKey.Wan2_2_T2V_14B)}
        assert pm_gen["tensor"] == {
            TensorKey.POSITIVE: TensorPoolKey(3, TensorKey.POSITIVE),
            TensorKey.NEGATIVE: TensorPoolKey(3, TensorKey.NEGATIVE),
        }

        # vae_dec(5): 入边 = loader_vae(2) → model, gen(4) → LATENTS
        pm_vae = calls[2].args[1]
        assert pm_vae["model"] == {ModelKey.VAE_WAN2_2: ModelPoolKey(2, ModelKey.VAE_WAN2_2)}
        assert pm_vae["tensor"] == {TensorKey.LATENTS: TensorPoolKey(4, TensorKey.LATENTS)}

        # save(6): 入边 = vae_dec(5) → VIDEO
        pm_save = calls[3].args[1]
        assert pm_save["model"] == {}
        assert pm_save["tensor"] == {TensorKey.VIDEO: TensorPoolKey(5, TensorKey.VIDEO)}

    def test_generate_build_context_receives_infer_task(self):
        """build_context() 接收等价的 InferTask。"""
        pipeline, _, ctx_builder = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        # build_context 应该被调用 4 次
        assert ctx_builder.build_context.call_count == 4

        # 验证第一个调用的 phase 参数是 InferTask
        first_phase = ctx_builder.build_context.call_args_list[0].args[0]
        assert isinstance(first_phase, InferTask)
        assert first_phase.node_type == NT.TEXT_ENCODE
        assert first_phase.model_key == ModelKey.T5TextEncoder

    def test_generate_tensor_scope_used(self):
        """generate() 在 tensor_scope 内执行。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        engine.tensor_scope.assert_called_once_with(keep=[TensorKey.VIDEO])

    def test_generate_prepare_tensors_called(self):
        """prepare_tensors() 为每个 infer 节点调用。"""
        pipeline, _, ctx_builder = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        assert ctx_builder.prepare_tensors.call_count == 4

    def test_generate_put_tensors_when_available(self):
        """prepare_tensors() 返回非 None 时调用 put_tensors()。"""
        pipeline, engine, ctx_builder = self._setup_pipeline_for_generate()
        # 第一个 infer 节点返回 tensor
        fake_tensors = {TensorKey.POSITIVE: "fake_tensor"}
        ctx_builder.prepare_tensors.side_effect = [fake_tensors, None, None, None]

        pipeline.generate("test prompt")

        engine.put_tensors.assert_called_once_with(fake_tensors)


# ── 条件跳过测试 ──────────────────────────────────────────────────────────


class TestConditionSkip:
    """条件跳过（check_condition）的行为。"""

    def _setup_pipeline_for_generate(self, *, condition_result=True):
        """构建一个带条件的 Pipeline。"""
        pipeline_def = _make_dag_pipeline_def(with_condition=True)
        engine = _make_mock_engine()
        ctx_builder = _make_mock_ctx_builder()
        ctx_builder.check_condition.return_value = condition_result
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder)
        pipeline._default_settings = _make_default_settings()
        return pipeline, engine, ctx_builder

    def test_generate_condition_skip(self):
        """condition 为 False 时跳过该节点。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate(condition_result=False)

        pipeline.generate("test prompt")

        # gen(4) 被跳过，只执行 3 个节点
        assert engine.run_infer_node.call_count == 3
        executed_ids = [c.args[0].node_id for c in engine.run_infer_node.call_args_list]
        assert 4 not in executed_ids
        # text_enc(3), vae_dec(5), save(6) 仍然执行
        assert 3 in executed_ids
        assert 5 in executed_ids
        assert 6 in executed_ids

    def test_generate_condition_pass(self):
        """condition 为 True 时正常执行。"""
        pipeline, engine, _ = self._setup_pipeline_for_generate(condition_result=True)

        pipeline.generate("test prompt")

        # 所有 4 个 infer 节点都执行
        assert engine.run_infer_node.call_count == 4

    def test_generate_condition_only_checked_for_conditional_nodes(self):
        """只有设置了 condition 的节点才调用 check_condition()。"""
        pipeline, _, ctx_builder = self._setup_pipeline_for_generate(condition_result=True)

        pipeline.generate("test prompt")

        # 只有 gen(4) 有 condition，所以 check_condition 只调用 1 次
        assert ctx_builder.check_condition.call_count == 1
        ctx_builder.check_condition.assert_called_once()
        call_args = ctx_builder.check_condition.call_args
        assert call_args.args[0] == "should_generate"


# ── clear() 测试 ──────────────────────────────────────────────────────────


class TestClear:
    """Pipeline.clear() 的行为。"""

    def test_clear_with_loaders(self):
        """有 loader 节点时调用 cleanup_distributed()。"""
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        pipeline = _make_pipeline(pipeline_def, engine)

        pipeline.clear()

        engine.cleanup_distributed.assert_called_once()

    def test_clear_empty_nodes_no_cleanup(self):
        """无 loader 节点时不调用 cleanup_distributed()。"""
        pipeline_def = PipelineDef(
            pipeline_key=PipelineKey.Wan2_2_T2V_14B,
            context_builder_cls=MagicMock,
        )
        engine = _make_mock_engine()
        pipeline = _make_pipeline(pipeline_def, engine)

        pipeline.clear()

        engine.cleanup_distributed.assert_not_called()


# ── _find_vae_model_key() 测试 ────────────────────────────────────────────


class TestFindVaeModelKey:
    """Pipeline._find_vae_model_key() 的行为。"""

    def test_finds_vae(self):
        """从 nodes 中找到 VAE key。"""
        pipeline_def = _make_dag_pipeline_def()
        engine = _make_mock_engine()
        pipeline = _make_pipeline(pipeline_def, engine)

        result = pipeline._find_vae_model_key()
        assert result == ModelKey.VAE_WAN2_2

    def test_no_vae_returns_none(self):
        """无 VAE 时返回 None。"""
        nodes = (NodeDef(node_id=0, is_loader=True, model_key=ModelKey.T5TextEncoder),)
        pipeline_def = PipelineDef(
            pipeline_key=PipelineKey.Wan2_2_T2V_14B,
            nodes=nodes,
            context_builder_cls=MagicMock,
        )
        engine = _make_mock_engine()
        pipeline = _make_pipeline(pipeline_def, engine)

        result = pipeline._find_vae_model_key()
        assert result is None
