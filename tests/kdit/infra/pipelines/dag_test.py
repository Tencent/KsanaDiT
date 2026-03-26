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

"""Pipeline DAG 遍历逻辑 + 拓扑排序 + pin 映射 单元测试。

测试 Pipeline 的 load_models() 和 generate() 的行为：
- 按拓扑序遍历 Loader / Infer 节点
- load_models() 调用 engine.run_node()
- generate() 调用 engine.run_node()
- input_pins 正确传递
- 条件跳过（check_condition）的行为
- build_context 接收 NodeDef（非 InferTask）

以及底层 DAG 工具函数：
- topo_sort() 拓扑排序
- compute_input_pins() pin 映射计算
"""

import unittest
from unittest.mock import MagicMock, patch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.nodes.core.node_types import IONodeType
from kdit.pipelines.dag import compute_input_pins, topo_sort
from kdit.pipelines.pipeline import (
    Pipeline,
    _node_def_display_name,
    _phase_display_name,
)
from kdit.pipelines.pipeline_def import Edge, NodeDef, PipelineDef
from kdit.pipelines.pipeline_key import PipelineKey
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool_key import TensorPoolKey

# ── 辅助：构建 mock 对象 ──────────────────────────────────────────────────


def _make_dag_pipeline_def(*, with_condition=False):
    """构建一个 DAG 模式的 PipelineDef。

    DAG 结构:
        loader_t5 ──model──> text_enc
        loader_dit ──model──> gen
        loader_vae ──model──> vae_dec
        text_enc ──POSITIVE──> gen
        text_enc ──NEGATIVE──> gen
        gen ──LATENTS──> vae_dec
        vae_dec ──VIDEO──> save

    Returns:
        (PipelineDef, nodes_dict) — nodes_dict 用于测试断言。
    """
    loader_t5 = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
    loader_dit = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.Wan2_2_T2V_14B)
    loader_vae = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.VAE_WAN2_2)
    text_enc = NodeDef(node_type=NT.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)
    gen = NodeDef(
        node_type=NT.GENERATE,
        model_key=ModelKey.Wan2_2_T2V_14B,
        condition="should_generate" if with_condition else None,
    )
    vae_dec = NodeDef(node_type=NT.VAE_DECODE, model_key=ModelKey.VAE_WAN2_2)
    save = NodeDef(node_type=NT.SAVE_VIDEO)

    nodes = (loader_t5, loader_dit, loader_vae, text_enc, gen, vae_dec, save)
    edges = (
        Edge(loader_t5.node_id, ModelKey.T5TextEncoder, text_enc.node_id, ModelKey.T5TextEncoder),
        Edge(loader_dit.node_id, ModelKey.Wan2_2_T2V_14B, gen.node_id, ModelKey.Wan2_2_T2V_14B),
        Edge(loader_vae.node_id, ModelKey.VAE_WAN2_2, vae_dec.node_id, ModelKey.VAE_WAN2_2),
        Edge(text_enc.node_id, TensorKey.POSITIVE, gen.node_id, TensorKey.POSITIVE),
        Edge(text_enc.node_id, TensorKey.NEGATIVE, gen.node_id, TensorKey.NEGATIVE),
        Edge(gen.node_id, TensorKey.LATENTS, vae_dec.node_id, TensorKey.LATENTS),
        Edge(vae_dec.node_id, TensorKey.VIDEO, save.node_id, TensorKey.VIDEO),
    )
    nodes_dict = {
        "loader_t5": loader_t5,
        "loader_dit": loader_dit,
        "loader_vae": loader_vae,
        "text_enc": text_enc,
        "gen": gen,
        "vae_dec": vae_dec,
        "save": save,
    }

    mock_ctx_cls = MagicMock
    pipeline_def = PipelineDef(
        pipeline_key=PipelineKey.Wan2_2_T2V_14B,
        nodes=nodes,
        edges=edges,
        keep_tensors=(TensorKey.VIDEO,),
        context_builder_cls=mock_ctx_cls,
    )
    return pipeline_def, nodes_dict


def _build_mock_node_outputs(nd):
    """为给定节点构建 mock output_pins。"""
    outputs = {}
    # loader 节点输出 model pin
    if nd.is_loader and nd.model_key:
        outputs[nd.model_key] = ModelPoolKey(nd.node_id, nd.model_key)
    # infer 节点根据类型输出 tensor pin
    if nd.node_type == NT.TEXT_ENCODE:
        outputs[TensorKey.POSITIVE] = TensorPoolKey(nd.node_id, TensorKey.POSITIVE)
        outputs[TensorKey.NEGATIVE] = TensorPoolKey(nd.node_id, TensorKey.NEGATIVE)
    elif nd.node_type == NT.GENERATE:
        outputs[TensorKey.LATENTS] = TensorPoolKey(nd.node_id, TensorKey.LATENTS)
    elif nd.node_type == NT.VAE_DECODE:
        outputs[TensorKey.VIDEO] = TensorPoolKey(nd.node_id, TensorKey.VIDEO)
    return outputs


def _make_mock_run_node(nodes_dict):
    """创建一个 mock_run_node side_effect 函数。"""
    all_outputs = {nd.node_id: _build_mock_node_outputs(nd) for nd in nodes_dict.values()}

    def _mock_run_node(node_def, input_pins, context):
        return dict(all_outputs.get(node_def.node_id, {}))

    return _mock_run_node, all_outputs


def _make_mock_engine(nodes_dict):
    """构建一个 mock Engine，模拟所有 Pipeline 需要的方法。"""
    engine = MagicMock()
    mock_tv = MagicMock()
    mock_tv.data = "fake_output"
    engine.get_tensor.return_value = mock_tv
    mock_run_node, _ = _make_mock_run_node(nodes_dict)
    engine.run_node.side_effect = mock_run_node
    return engine


def _make_mock_ctx_builder():
    """构建一个 mock ContextBuilder。"""
    ctx_builder = MagicMock()
    ctx_builder.build_loader_kwargs.return_value = {"model_path": "/fake/path"}
    ctx_builder.build_context.return_value = NodeContext(prompt="test")
    ctx_builder.check_condition.return_value = True
    ctx_builder.post_process.side_effect = lambda output, inputs: output
    ctx_builder.prepare_generate_inputs.return_value = None
    ctx_builder.resolve_model_paths.return_value = ("/fake", "/fake/text", "/fake/vae")
    ctx_builder.resolve_lora_config.return_value = None
    return ctx_builder


def _make_pipeline(pipeline_def, engine=None, ctx_builder=None, nodes_dict=None):
    """构建 Pipeline 实例，注入 mock 依赖。"""
    engine = engine or _make_mock_engine(nodes_dict or {})
    pipeline = Pipeline(pipeline_def, engine)
    if ctx_builder:
        pipeline._ctx_builder = ctx_builder
    return pipeline


def _make_default_settings():
    """构建一个合法的 _default_settings mock，使 _prepare_configs 不报错。"""
    settings = MagicMock()
    settings.sample_config.steps = None
    settings.sample_config.shift = None
    settings.sample_config.denoise = None
    settings.sample_config.cfg_scale = None
    settings.sample_config.solver = None
    settings.runtime_config.target_size = None
    settings.runtime_config.frame_num = None
    settings.cache = None
    return settings


# ── 显示名称测试 ──────────────────────────────────────────────────────────


class TestNodeDefDisplayName:
    """_node_def_display_name() 和 _phase_display_name() 对 NodeDef 的处理。"""

    def test_loader_node_def(self):
        """Loader NodeDef 显示为 LOAD(model_name)。"""
        nd = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        assert _node_def_display_name(nd) == "LOAD(T5TextEncoder)"

    def test_loader_node_def_no_model_key(self):
        """Loader NodeDef 无 model_key 时显示 LOAD(UNKNOWN)。"""
        nd = NodeDef(node_type=IONodeType.LOAD_MODEL)
        assert _node_def_display_name(nd) == "LOAD(UNKNOWN)"

    def test_infer_node_def_with_model(self):
        """Infer NodeDef 有 model_key 时显示 TYPE(model_name)。"""
        nd = NodeDef(node_type=NT.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)
        assert _node_def_display_name(nd) == "GENERATE(Wan2_2_T2V_14B)"

    def test_infer_node_def_no_model(self):
        """Infer NodeDef 无 model_key 时只显示 TYPE。"""
        nd = NodeDef(node_type=NT.SAVE_VIDEO)
        assert _node_def_display_name(nd) == "SAVE_VIDEO"

    def test_infer_node_def_no_model_save_image(self):
        """Infer NodeDef 无 model_key 时只显示 TYPE（SAVE_IMAGE 示例）。"""
        nd = NodeDef(node_type=NT.SAVE_IMAGE)
        assert _node_def_display_name(nd) == "SAVE_IMAGE"

    def test_phase_display_name_dispatches_to_node_def(self):
        """_phase_display_name() 对 NodeDef 类型正确分发。"""
        nd = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        assert _phase_display_name(nd) == "LOAD(T5TextEncoder)"


# ── load_models() 测试 ────────────────────────────────────────────────────


class TestLoadModels:
    """Pipeline.load_models() 的行为。"""

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_calls_run_node_for_loaders_only(self, mock_settings):
        """load_models() 只对 is_loader=True 的节点调用 engine.run_node()。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)

        pipeline.load_models("/fake/path")

        assert engine.run_node.call_count == 3

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_topo_order(self, mock_settings):
        """loader 按拓扑序执行。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)

        pipeline.load_models("/fake/path")

        call_node_defs = [c.args[0] for c in engine.run_node.call_args_list]
        assert all(n.is_loader for n in call_node_defs)
        # 按拓扑序: loader_t5, loader_dit, loader_vae
        assert [n.node_id for n in call_node_defs] == [
            nd["loader_t5"].node_id,
            nd["loader_dit"].node_id,
            nd["loader_vae"].node_id,
        ]

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_input_pins_passed(self, mock_settings):
        """input_pins 正确传递给 engine.run_node()。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)

        pipeline.load_models("/fake/path")

        for c in engine.run_node.call_args_list:
            input_pins = c.args[1]
            assert input_pins == {}

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_context_has_loader_kwargs(self, mock_settings):
        """context.metadata 包含 build_loader_kwargs() 的结果。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        ctx_builder.build_loader_kwargs.return_value = {"model_path": "/test/model"}
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)

        pipeline.load_models("/fake/path")

        for c in engine.run_node.call_args_list:
            context = c.args[2]
            assert isinstance(context, NodeContext)
            assert context.metadata == {"model_path": "/test/model"}

    @patch("kdit.pipelines.pipeline.load_default_settings")
    def test_load_build_loader_kwargs_called_per_loader(self, mock_settings):
        """build_loader_kwargs() 为每个 loader 调用一次。"""
        mock_settings.return_value = _make_default_settings()
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)

        pipeline.load_models("/fake/path")

        assert ctx_builder.build_loader_kwargs.call_count == 3
        called_keys = [c.args[0] for c in ctx_builder.build_loader_kwargs.call_args_list]
        assert called_keys == [
            ModelKey.T5TextEncoder,
            ModelKey.Wan2_2_T2V_14B,
            ModelKey.VAE_WAN2_2,
        ]


# ── generate() 测试 ───────────────────────────────────────────────────────


class TestGenerate:
    """Pipeline.generate() 的行为。"""

    def _setup_pipeline_for_generate(self, *, with_condition=False):
        """构建一个可以调用 generate() 的 Pipeline。"""
        pipeline_def, nd = _make_dag_pipeline_def(with_condition=with_condition)
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)
        pipeline._default_settings = _make_default_settings()
        # 预填充 loader 阶段的 output_pins
        loader_ids = {nd["loader_t5"].node_id, nd["loader_dit"].node_id, nd["loader_vae"].node_id}
        all_outputs = {n.node_id: _build_mock_node_outputs(n) for n in nd.values()}
        pipeline._loader_outputs = {nid: dict(pins) for nid, pins in all_outputs.items() if nid in loader_ids}
        return pipeline, engine, ctx_builder, nd

    def test_generate_calls_run_node_for_infer_only(self):
        """generate() 只对 is_loader=False 的节点调用 engine.run_node()。"""
        pipeline, engine, _, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        assert engine.run_node.call_count == 4

    def test_generate_topo_order(self):
        """infer 节点按拓扑序执行。"""
        pipeline, engine, _, nd = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        call_node_defs = [c.args[0] for c in engine.run_node.call_args_list]
        assert all(not n.is_loader for n in call_node_defs)
        # 拓扑序: text_enc → gen → vae_dec → save
        assert [n.node_id for n in call_node_defs] == [
            nd["text_enc"].node_id,
            nd["gen"].node_id,
            nd["vae_dec"].node_id,
            nd["save"].node_id,
        ]

    def test_generate_input_pins_correct(self):
        """infer 节点的 input_pins 正确。"""
        pipeline, engine, _, nd = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        calls = engine.run_node.call_args_list

        # text_enc: 入边 = loader_t5 → model
        ip_text = calls[0].args[1]
        assert ip_text[ModelKey.T5TextEncoder] == ModelPoolKey(nd["loader_t5"].node_id, ModelKey.T5TextEncoder)
        assert len(ip_text) == 1

        # gen: 入边 = loader_dit → model, text_enc → POSITIVE/NEGATIVE
        ip_gen = calls[1].args[1]
        assert ip_gen[ModelKey.Wan2_2_T2V_14B] == ModelPoolKey(nd["loader_dit"].node_id, ModelKey.Wan2_2_T2V_14B)
        assert ip_gen[TensorKey.POSITIVE] == TensorPoolKey(nd["text_enc"].node_id, TensorKey.POSITIVE)
        assert ip_gen[TensorKey.NEGATIVE] == TensorPoolKey(nd["text_enc"].node_id, TensorKey.NEGATIVE)
        assert len(ip_gen) == 3

        # vae_dec: 入边 = loader_vae → model, gen → LATENTS
        ip_vae = calls[2].args[1]
        assert ip_vae[ModelKey.VAE_WAN2_2] == ModelPoolKey(nd["loader_vae"].node_id, ModelKey.VAE_WAN2_2)
        assert ip_vae[TensorKey.LATENTS] == TensorPoolKey(nd["gen"].node_id, TensorKey.LATENTS)
        assert len(ip_vae) == 2

        # save: 入边 = vae_dec → VIDEO
        ip_save = calls[3].args[1]
        assert ip_save[TensorKey.VIDEO] == TensorPoolKey(nd["vae_dec"].node_id, TensorKey.VIDEO)
        assert len(ip_save) == 1

    def test_generate_build_context_receives_node_def(self):
        """build_context() 接收 NodeDef 实例。"""
        pipeline, _, ctx_builder, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        assert ctx_builder.build_context.call_count == 4

        first_arg = ctx_builder.build_context.call_args_list[0].args[0]
        assert isinstance(first_arg, NodeDef)
        assert first_arg.node_type == NT.TEXT_ENCODE
        assert first_arg.model_key == ModelKey.T5TextEncoder

    def test_generate_clear_all_tensors_called(self):
        """generate() 在 finally 中调用 clear_all_tensors()。"""
        pipeline, engine, _, _ = self._setup_pipeline_for_generate()

        pipeline.generate("test prompt")

        engine.clear_all_tensors.assert_called_once()

    def test_generate_extra_inputs_passed_to_prepare(self):
        """extra_inputs 参数传递给 prepare_generate_inputs()。"""
        pipeline, _, ctx_builder, _ = self._setup_pipeline_for_generate()

        from kdit.pipelines.extra_inputs import ExtraInputs

        extra = ExtraInputs()
        pipeline.generate("test prompt", extra_inputs=extra)

        call_args = ctx_builder.prepare_generate_inputs.call_args
        assert call_args.args[1] is extra


# ── 条件跳过测试 ──────────────────────────────────────────────────────────


class TestConditionSkip:
    """条件跳过（check_condition）的行为。"""

    def _setup_pipeline_for_generate(self, *, condition_result=True):
        """构建一个带条件的 Pipeline。"""
        pipeline_def, nd = _make_dag_pipeline_def(with_condition=True)
        engine = _make_mock_engine(nd)
        ctx_builder = _make_mock_ctx_builder()
        ctx_builder.check_condition.return_value = condition_result
        pipeline = _make_pipeline(pipeline_def, engine, ctx_builder, nd)
        pipeline._default_settings = _make_default_settings()
        loader_ids = {nd["loader_t5"].node_id, nd["loader_dit"].node_id, nd["loader_vae"].node_id}
        all_outputs = {n.node_id: _build_mock_node_outputs(n) for n in nd.values()}
        pipeline._loader_outputs = {nid: dict(pins) for nid, pins in all_outputs.items() if nid in loader_ids}
        return pipeline, engine, ctx_builder, nd

    def test_generate_condition_skip(self):
        """condition 为 False 时跳过该节点。"""
        pipeline, engine, _, nd = self._setup_pipeline_for_generate(condition_result=False)

        pipeline.generate("test prompt")

        # gen 被跳过，只执行 3 个节点
        assert engine.run_node.call_count == 3
        executed_ids = [c.args[0].node_id for c in engine.run_node.call_args_list]
        assert nd["gen"].node_id not in executed_ids
        # text_enc, vae_dec, save 仍然执行
        assert nd["text_enc"].node_id in executed_ids
        assert nd["vae_dec"].node_id in executed_ids
        assert nd["save"].node_id in executed_ids

    def test_generate_condition_pass(self):
        """condition 为 True 时正常执行。"""
        pipeline, engine, _, _ = self._setup_pipeline_for_generate(condition_result=True)

        pipeline.generate("test prompt")

        assert engine.run_node.call_count == 4

    def test_generate_condition_only_checked_for_conditional_nodes(self):
        """只有设置了 condition 的节点才调用 check_condition()。"""
        pipeline, _, ctx_builder, _ = self._setup_pipeline_for_generate(condition_result=True)

        pipeline.generate("test prompt")

        assert ctx_builder.check_condition.call_count == 1
        ctx_builder.check_condition.assert_called_once()
        call_args = ctx_builder.check_condition.call_args
        assert call_args.args[0] == "should_generate"


# ── clear() 测试 ──────────────────────────────────────────────────────────


class TestClear:
    """Pipeline.clear() 的行为。"""

    def test_clear_with_loaders(self):
        """有 loader 节点时调用 cleanup_distributed()。"""
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        pipeline = _make_pipeline(pipeline_def, engine, nodes_dict=nd)

        pipeline.clear()

        engine.cleanup_distributed.assert_called_once()

    def test_clear_empty_nodes_no_cleanup(self):
        """无 loader 节点时不调用 cleanup_distributed()。"""
        pipeline_def = PipelineDef(
            pipeline_key=PipelineKey.Wan2_2_T2V_14B,
            context_builder_cls=MagicMock,
        )
        engine = _make_mock_engine({})
        pipeline = _make_pipeline(pipeline_def, engine, nodes_dict={})

        pipeline.clear()

        engine.cleanup_distributed.assert_not_called()


# ── _find_vae_model_key() 测试 ────────────────────────────────────────────


class TestFindVaeModelKey:
    """Pipeline._find_vae_model_key() 的行为。"""

    def test_finds_vae(self):
        """从 nodes 中找到 VAE key。"""
        pipeline_def, nd = _make_dag_pipeline_def()
        engine = _make_mock_engine(nd)
        pipeline = _make_pipeline(pipeline_def, engine, nodes_dict=nd)

        result = pipeline._find_vae_model_key()
        assert result == ModelKey.VAE_WAN2_2

    def test_no_vae_returns_none(self):
        """无 VAE 时返回 None。"""
        single_node = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        nodes = (single_node,)
        pipeline_def = PipelineDef(
            pipeline_key=PipelineKey.Wan2_2_T2V_14B,
            nodes=nodes,
            context_builder_cls=MagicMock,
        )
        engine = _make_mock_engine({})
        pipeline = _make_pipeline(pipeline_def, engine, nodes_dict={})

        result = pipeline._find_vae_model_key()
        assert result is None


# ── topo_sort() 测试 ──────────────────────────────────────────────────────


class TestTopoSort(unittest.TestCase):
    """topo_sort() 拓扑排序测试。"""

    def test_linear_dag(self):
        """线性 DAG: A → B → C 拓扑排序结果正确。"""
        a = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_type=NT.TEXT_ENCODE)
        c = NodeDef(node_type=NT.SAVE_VIDEO)

        nodes = (a, b, c)
        edges = (
            Edge(a.node_id, ModelKey.T5TextEncoder, b.node_id, ModelKey.T5TextEncoder),
            Edge(b.node_id, TensorKey.POSITIVE, c.node_id, TensorKey.POSITIVE),
        )

        result = topo_sort(nodes, edges)
        self.assertEqual([n.node_id for n in result], [a.node_id, b.node_id, c.node_id])

    def test_diamond_dag(self):
        """菱形 DAG: A→B, A→C, B→D, C→D 拓扑排序正确。"""
        a = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_type=NT.TEXT_ENCODE)
        c = NodeDef(node_type=NT.VAE_DECODE)
        d = NodeDef(node_type=NT.SAVE_VIDEO)

        nodes = (a, b, c, d)
        edges = (
            Edge(a.node_id, TensorKey.POSITIVE, b.node_id, TensorKey.POSITIVE),
            Edge(a.node_id, TensorKey.NEGATIVE, c.node_id, TensorKey.NEGATIVE),
            Edge(b.node_id, TensorKey.LATENTS, d.node_id, TensorKey.LATENTS),
            Edge(c.node_id, TensorKey.VIDEO, d.node_id, TensorKey.VIDEO),
        )

        result = topo_sort(nodes, edges)
        ids = [n.node_id for n in result]

        # A 必须在 B 和 C 之前
        self.assertLess(ids.index(a.node_id), ids.index(b.node_id))
        self.assertLess(ids.index(a.node_id), ids.index(c.node_id))
        # B 和 C 必须在 D 之前
        self.assertLess(ids.index(b.node_id), ids.index(d.node_id))
        self.assertLess(ids.index(c.node_id), ids.index(d.node_id))

    def test_multiple_edges_same_pair(self):
        """同一对 src→dst 有多条边时，依赖只算一次。"""
        a = NodeDef(node_type=NT.TEXT_ENCODE)
        b = NodeDef(node_type=NT.GENERATE)

        nodes = (a, b)
        edges = (
            Edge(a.node_id, TensorKey.POSITIVE, b.node_id, TensorKey.POSITIVE),
            Edge(a.node_id, TensorKey.NEGATIVE, b.node_id, TensorKey.NEGATIVE),
        )

        result = topo_sort(nodes, edges)
        self.assertEqual([n.node_id for n in result], [a.node_id, b.node_id])

    def test_isolated_nodes(self):
        """无边的独立节点也能排序。"""
        a = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        b = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.VAE_WAN2_2)

        nodes = (a, b)
        edges = ()

        result = topo_sort(nodes, edges)
        self.assertEqual(len(result), 2)
        # 按 node_id 排序
        self.assertEqual([n.node_id for n in result], [a.node_id, b.node_id])

    def test_cycle_detection(self):
        """有环时抛出 ValueError。"""
        a = NodeDef(node_type=NT.TEXT_ENCODE)
        b = NodeDef(node_type=NT.GENERATE)

        nodes = (a, b)
        edges = (
            Edge(a.node_id, TensorKey.POSITIVE, b.node_id, TensorKey.POSITIVE),
            Edge(b.node_id, TensorKey.NEGATIVE, a.node_id, TensorKey.NEGATIVE),
        )

        with self.assertRaises(ValueError, msg="cycle"):
            topo_sort(nodes, edges)

    def test_loaders_first(self):
        """Loader 节点（入度 0）排在 Infer 节点前面。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        infer = NodeDef(node_type=NT.TEXT_ENCODE)

        nodes = (infer, loader)  # 故意反序
        edges = (Edge(loader.node_id, ModelKey.T5TextEncoder, infer.node_id, ModelKey.T5TextEncoder),)

        result = topo_sort(nodes, edges)
        self.assertEqual(result[0].node_id, loader.node_id)
        self.assertEqual(result[1].node_id, infer.node_id)


# ── compute_input_pins() 测试 ─────────────────────────────────────────────


class TestComputeInputPins(unittest.TestCase):
    """compute_input_pins() 测试。"""

    def test_tensor_mapping(self):
        """正确计算 tensor 映射。"""
        src = NodeDef(node_type=NT.TEXT_ENCODE)
        node = NodeDef(node_type=NT.GENERATE)
        edges = (
            Edge(src.node_id, TensorKey.POSITIVE, node.node_id, TensorKey.POSITIVE),
            Edge(src.node_id, TensorKey.NEGATIVE, node.node_id, TensorKey.NEGATIVE),
        )

        mapping = compute_input_pins(node, edges)

        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[TensorKey.POSITIVE], TensorPoolKey(src.node_id, TensorKey.POSITIVE))
        self.assertEqual(mapping[TensorKey.NEGATIVE], TensorPoolKey(src.node_id, TensorKey.NEGATIVE))

    def test_model_mapping(self):
        """正确计算 model 映射。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        node = NodeDef(node_type=NT.TEXT_ENCODE)
        edges = (Edge(loader.node_id, ModelKey.T5TextEncoder, node.node_id, ModelKey.T5TextEncoder),)

        mapping = compute_input_pins(node, edges)

        self.assertEqual(len(mapping), 1)
        self.assertEqual(mapping[ModelKey.T5TextEncoder], ModelPoolKey(loader.node_id, ModelKey.T5TextEncoder))

    def test_mixed_mapping(self):
        """同时有 tensor 和 model 映射。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.Wan2_2_T2V_14B)
        text_enc = NodeDef(node_type=NT.TEXT_ENCODE)
        node = NodeDef(node_type=NT.GENERATE)
        # 另一个不相关的 loader
        other_loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        other_infer = NodeDef(node_type=NT.TEXT_ENCODE)

        edges = (
            Edge(loader.node_id, ModelKey.Wan2_2_T2V_14B, node.node_id, ModelKey.Wan2_2_T2V_14B),
            Edge(text_enc.node_id, TensorKey.POSITIVE, node.node_id, TensorKey.POSITIVE),
            Edge(text_enc.node_id, TensorKey.NEGATIVE, node.node_id, TensorKey.NEGATIVE),
            # 不相关的边（目标不是 node）
            Edge(other_loader.node_id, ModelKey.T5TextEncoder, other_infer.node_id, ModelKey.T5TextEncoder),
        )

        mapping = compute_input_pins(node, edges)

        # 3 个 pin：1 model + 2 tensor
        self.assertEqual(len(mapping), 3)
        self.assertEqual(
            mapping[ModelKey.Wan2_2_T2V_14B],
            ModelPoolKey(loader.node_id, ModelKey.Wan2_2_T2V_14B),
        )

    def test_no_edges_for_node(self):
        """节点没有入边时返回空映射。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.T5TextEncoder)
        infer = NodeDef(node_type=NT.TEXT_ENCODE)
        edges = (Edge(loader.node_id, ModelKey.T5TextEncoder, infer.node_id, ModelKey.T5TextEncoder),)

        mapping = compute_input_pins(loader, edges)

        self.assertEqual(mapping, {})

    def test_dynamic_mode_from_all_outputs(self):
        """动态模式：从 all_outputs 查找实际 PoolKey。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.Wan2_2_T2V_14B)
        text_enc = NodeDef(node_type=NT.TEXT_ENCODE)
        node = NodeDef(node_type=NT.GENERATE)
        edges = (
            Edge(loader.node_id, ModelKey.Wan2_2_T2V_14B, node.node_id, ModelKey.Wan2_2_T2V_14B),
            Edge(text_enc.node_id, TensorKey.POSITIVE, node.node_id, TensorKey.POSITIVE),
        )
        all_outputs = {
            loader.node_id: {ModelKey.Wan2_2_T2V_14B: ModelPoolKey(loader.node_id, ModelKey.Wan2_2_T2V_14B)},
            text_enc.node_id: {TensorKey.POSITIVE: TensorPoolKey(text_enc.node_id, TensorKey.POSITIVE)},
        }

        mapping = compute_input_pins(node, edges, all_outputs)

        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[ModelKey.Wan2_2_T2V_14B], ModelPoolKey(loader.node_id, ModelKey.Wan2_2_T2V_14B))
        self.assertEqual(mapping[TensorKey.POSITIVE], TensorPoolKey(text_enc.node_id, TensorKey.POSITIVE))

    def test_dynamic_mode_missing_upstream_skips_pin(self):
        """动态模式：上游节点不在 all_outputs 中时跳过该 pin。"""
        src = NodeDef(node_type=NT.GENERATE)
        node = NodeDef(node_type=NT.VAE_DECODE)
        edges = (Edge(src.node_id, TensorKey.LATENTS, node.node_id, TensorKey.LATENTS),)
        all_outputs: dict = {}  # 上游不存在（被跳过）

        mapping = compute_input_pins(node, edges, all_outputs)

        self.assertEqual(mapping, {})

    def test_dynamic_mode_matches_static_mode(self):
        """动态模式与静态模式产生相同结果（当 output_pins 结构一致时）。"""
        loader = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=ModelKey.Wan2_2_T2V_14B)
        text_enc = NodeDef(node_type=NT.TEXT_ENCODE)
        node = NodeDef(node_type=NT.GENERATE)
        edges = (
            Edge(loader.node_id, ModelKey.Wan2_2_T2V_14B, node.node_id, ModelKey.Wan2_2_T2V_14B),
            Edge(text_enc.node_id, TensorKey.POSITIVE, node.node_id, TensorKey.POSITIVE),
            Edge(text_enc.node_id, TensorKey.NEGATIVE, node.node_id, TensorKey.NEGATIVE),
        )
        all_outputs = {
            loader.node_id: {ModelKey.Wan2_2_T2V_14B: ModelPoolKey(loader.node_id, ModelKey.Wan2_2_T2V_14B)},
            text_enc.node_id: {
                TensorKey.POSITIVE: TensorPoolKey(text_enc.node_id, TensorKey.POSITIVE),
                TensorKey.NEGATIVE: TensorPoolKey(text_enc.node_id, TensorKey.NEGATIVE),
            },
        }

        static_mapping = compute_input_pins(node, edges)
        dynamic_mapping = compute_input_pins(node, edges, all_outputs)

        self.assertEqual(static_mapping, dynamic_mapping)


if __name__ == "__main__":
    unittest.main()
