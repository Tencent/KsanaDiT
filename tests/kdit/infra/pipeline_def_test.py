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

"""PipelineDefBuilder 构建与校验 / PipelineDef 注册表 / PipelineDef 结构校验 单元测试。"""

import unittest

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.context_builder import ContextBuilder
from kdit.pipelines.pin_ref import NodeRef
from kdit.pipelines.pipeline_def import (
    PipelineDef,
    PipelineDefBuilder,
    _InferTaskChain,
    get_pipeline_def,
)
from kdit.pipelines.pipeline_key import PipelineKey
from kdit.tensor import TensorKey

# ── 测试用 ContextBuilder ────────────────────────────────────────────────


class _DummyContextBuilder(ContextBuilder):
    """最小可用的 ContextBuilder — 用于 Builder 测试。"""

    def build_context(self, phase, inputs):
        return NodeContext()


# ── PipelineDefBuilder 旧模式测试 ────────────────────────────────────────


class TestPipelineDefBuilder(unittest.TestCase):
    """PipelineDefBuilder 链式构建与校验（旧线性模式）。"""

    def test_basic_build(self):
        """基本构建 — 3 个 load + 4 个 infer + context_builder。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .load(ModelKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .add_infer(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)
            .add_infer(NT.VAE_DECODE, ModelKey.VAE_WAN2_2)
            .add_infer(NT.SAVE_VIDEO)
            .keep_tensors(TensorKey.VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        self.assertIsInstance(pipeline_def, PipelineDef)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(len(pipeline_def.load_phases), 3)
        self.assertEqual(len(pipeline_def.infer_phases), 4)
        self.assertEqual(pipeline_def.keep_tensors, (TensorKey.VIDEO,))
        self.assertIs(pipeline_def.context_builder_cls, _DummyContextBuilder)

    def test_load_phases_order(self):
        """load_phases 保持声明顺序。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .load(ModelKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        keys = [lp.model_key for lp in pipeline_def.load_phases]
        self.assertEqual(keys, [ModelKey.T5TextEncoder, ModelKey.Wan2_2_T2V_14B, ModelKey.VAE_WAN2_2])

    def test_infer_phases_order(self):
        """infer_phases 保持声明顺序。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .load(ModelKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .add_infer(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)
            .add_infer(NT.VAE_DECODE, ModelKey.VAE_WAN2_2)
            .add_infer(NT.SAVE_VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        types = [ip.node_type for ip in pipeline_def.infer_phases]
        self.assertEqual(types, [NT.TEXT_ENCODE, NT.GENERATE, NT.VAE_DECODE, NT.SAVE_VIDEO])

    def test_when_condition(self):
        """add_infer().when() 设置条件。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2)
            .when("has_test_input")
            .add_infer(NT.SAVE_VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        # VAE_ENCODE_SPATIAL 有条件
        self.assertEqual(pipeline_def.infer_phases[0].condition, "has_test_input")
        # SAVE_VIDEO 无条件
        self.assertIsNone(pipeline_def.infer_phases[1].condition)

    def test_when_chain_returns_builder(self):
        """add_infer().when() 返回 PipelineDefBuilder，可继续链式。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)
        builder.load(ModelKey.VAE_WAN2_2)
        chain = builder.add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2)
        self.assertIsInstance(chain, _InferTaskChain)
        result = chain.when("has_test_input")
        self.assertIs(result, builder)

    def test_chain_proxy_without_when(self):
        """不调用 .when() 时，_InferTaskChain 代理到 builder。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .keep_tensors(TensorKey.VIDEO)  # 通过 __getattr__ 代理
            .context_builder(_DummyContextBuilder)
            .build()
        )
        self.assertEqual(pipeline_def.keep_tensors, (TensorKey.VIDEO,))

    def test_save_node_no_model_key(self):
        """SaveNode 的 model_key 为 None。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.SAVE_VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )
        self.assertIsNone(pipeline_def.infer_phases[0].model_key)

    def test_frozen_dataclass(self):
        """PipelineDef 是 frozen dataclass — 不可修改。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.SAVE_VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )
        with self.assertRaises(AttributeError):
            pipeline_def.pipeline_key = PipelineKey.QwenImage_T2I

    # ── 校验错误 ──

    def test_missing_context_builder(self):
        """缺少 context_builder 时 build() 报错。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B).load(ModelKey.VAE_WAN2_2).add_infer(NT.SAVE_VIDEO)
        with self.assertRaises(ValueError, msg="context_builder_cls is required"):
            builder.build()

    def test_missing_load_phases(self):
        """缺少 load phase 时 build() 报错。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_infer(NT.SAVE_VIDEO)
        builder.context_builder(_DummyContextBuilder)
        with self.assertRaises(ValueError, msg="At least one load phase"):
            builder.build()

    def test_missing_infer_phases(self):
        """缺少 infer phase 时 build() 报错。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.load(ModelKey.VAE_WAN2_2)
        builder.context_builder(_DummyContextBuilder)
        with self.assertRaises(ValueError, msg="At least one infer phase"):
            builder.build()

    def test_invalid_model_key_reference(self):
        """infer_phase 引用未声明的 model_key 时 build() 报错。"""
        builder = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)  # 未在 load 中声明
            .context_builder(_DummyContextBuilder)
        )
        with self.assertRaises(ValueError, msg="not declared in any LoadTask"):
            builder.build()


# ── PipelineDefBuilder DAG 模式测试 ──────────────────────────────────────


class TestPipelineDefBuilderDAG(unittest.TestCase):
    """PipelineDefBuilder DAG 模式构建与校验。"""

    def test_add_loader_returns_node_ref(self):
        """add_loader() 返回 NodeRef，node_id 从 0 开始自增。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        ref0 = builder.add_loader(ModelKey.T5TextEncoder)
        ref1 = builder.add_loader(ModelKey.Wan2_2_T2V_14B)

        self.assertIsInstance(ref0, NodeRef)
        self.assertIsInstance(ref1, NodeRef)
        self.assertEqual(ref0.node_id, 0)
        self.assertEqual(ref1.node_id, 1)

    def test_add_infer_dag_mode_returns_node_ref(self):
        """add_infer() 在 DAG 模式下返回 NodeRef。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)  # 切换到 DAG 模式
        ref = builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)

        self.assertIsInstance(ref, NodeRef)
        self.assertEqual(ref.node_id, 1)

    def test_alloc_node_id_increments(self):
        """_alloc_node_id() 从 0 开始递增。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(builder._alloc_node_id(), 0)
        self.assertEqual(builder._alloc_node_id(), 1)
        self.assertEqual(builder._alloc_node_id(), 2)

    def test_connect_format2_pinref(self):
        """connect() 格式 2（PinRef 引用）正确添加 Edge。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        t5 = builder.add_loader(ModelKey.T5TextEncoder)
        enc = builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)

        builder.connect(
            (t5.T5TextEncoder, enc.T5TextEncoder),
        )

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        self.assertEqual(len(pipeline_def.edges), 1)

        edge = pipeline_def.edges[0]
        self.assertEqual(edge.src_node_id, 0)
        self.assertEqual(edge.src_pin, ModelKey.T5TextEncoder)
        self.assertEqual(edge.dst_node_id, 1)
        self.assertEqual(edge.dst_pin, ModelKey.T5TextEncoder)
        self.assertEqual(edge.edge_type, "model")

    def test_connect_format1_infer_node_type(self):
        """connect() 格式 1（InferNodeType 引用）正确添加 Edge。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        builder.add_infer(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)

        builder.connect(
            (NT.TEXT_ENCODE, TensorKey.POSITIVE, NT.GENERATE, TensorKey.POSITIVE),
        )

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        self.assertEqual(len(pipeline_def.edges), 1)

        edge = pipeline_def.edges[0]
        self.assertEqual(edge.src_pin, TensorKey.POSITIVE)
        self.assertEqual(edge.dst_pin, TensorKey.POSITIVE)
        self.assertEqual(edge.edge_type, "tensor")

    def test_connect_one_to_many(self):
        """connect() 一对多 (src, [dst1, dst2]) 生成多条 Edge。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)  # 切换到 DAG 模式
        enc = builder.add_infer(NT.TEXT_ENCODE)
        gen = builder.add_infer(NT.GENERATE)
        dec = builder.add_infer(NT.VAE_DECODE)

        builder.connect(
            (enc.POSITIVE, [gen.POSITIVE, dec.POSITIVE]),
        )

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        self.assertEqual(len(pipeline_def.edges), 2)

        # 第一条边: enc → gen (node_id 1 → 2, 因为 loader 是 0)
        self.assertEqual(pipeline_def.edges[0].src_node_id, 1)
        self.assertEqual(pipeline_def.edges[0].dst_node_id, 2)
        # 第二条边: enc → dec (node_id 1 → 3)
        self.assertEqual(pipeline_def.edges[1].src_node_id, 1)
        self.assertEqual(pipeline_def.edges[1].dst_node_id, 3)

    def test_build_dag_returns_frozen_pipeline_def(self):
        """build() 返回 frozen PipelineDef，nodes 和 edges 正确。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        t5 = builder.add_loader(ModelKey.T5TextEncoder)
        enc = builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        save = builder.add_infer(NT.SAVE_VIDEO)

        builder.connect(
            (t5.T5TextEncoder, enc.T5TextEncoder),
            (enc.VIDEO, save.VIDEO),
        )

        pipeline_def = builder.keep_tensors(TensorKey.VIDEO).context_builder(_DummyContextBuilder).build()

        self.assertIsInstance(pipeline_def, PipelineDef)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(len(pipeline_def.nodes), 3)
        self.assertEqual(len(pipeline_def.edges), 2)
        self.assertEqual(pipeline_def.keep_tensors, (TensorKey.VIDEO,))
        self.assertIs(pipeline_def.context_builder_cls, _DummyContextBuilder)

        # DAG 模式下旧字段为空
        self.assertEqual(pipeline_def.load_phases, ())
        self.assertEqual(pipeline_def.infer_phases, ())

        # frozen
        with self.assertRaises(AttributeError):
            pipeline_def.pipeline_key = PipelineKey.QwenImage_T2I

    def test_legacy_mode_still_works(self):
        """旧模式 .load() + .add_infer() 仍然可用（向后兼容）。"""
        pipeline_def = (
            PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
            .load(ModelKey.T5TextEncoder)
            .load(ModelKey.Wan2_2_T2V_14B)
            .load(ModelKey.VAE_WAN2_2)
            .add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
            .add_infer(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)
            .add_infer(NT.VAE_DECODE, ModelKey.VAE_WAN2_2)
            .add_infer(NT.SAVE_VIDEO)
            .keep_tensors(TensorKey.VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        # 旧模式字段有值
        self.assertEqual(len(pipeline_def.load_phases), 3)
        self.assertEqual(len(pipeline_def.infer_phases), 4)
        # DAG 字段为空
        self.assertEqual(pipeline_def.nodes, ())
        self.assertEqual(pipeline_def.edges, ())

    def test_node_defs_structure(self):
        """NodeDef 结构正确 — loader 和 infer 的字段。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        builder.add_infer(NT.SAVE_VIDEO)

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()

        # Loader node
        loader = pipeline_def.nodes[0]
        self.assertTrue(loader.is_loader)
        self.assertEqual(loader.model_key, ModelKey.T5TextEncoder)
        self.assertIsNone(loader.node_type)

        # Infer node with model
        infer = pipeline_def.nodes[1]
        self.assertFalse(infer.is_loader)
        self.assertEqual(infer.node_type, NT.TEXT_ENCODE)
        self.assertEqual(infer.model_key, ModelKey.T5TextEncoder)

        # Infer node without model
        save = pipeline_def.nodes[2]
        self.assertFalse(save.is_loader)
        self.assertEqual(save.node_type, NT.SAVE_VIDEO)
        self.assertIsNone(save.model_key)

    # ── DAG 校验错误 ──

    def test_type_mismatch_raises(self):
        """TensorKey 连 ModelKey 时 connect() 抛出 TypeError。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)  # 切换到 DAG 模式
        enc = builder.add_infer(NT.TEXT_ENCODE)
        gen = builder.add_infer(NT.GENERATE)

        with self.assertRaises(TypeError, msg="type mismatch"):
            builder.connect(
                (enc.POSITIVE, gen.T5TextEncoder),  # TensorKey → ModelKey
            )

    def test_duplicate_input_raises(self):
        """同一 dst pin 有两条入边时 build() 抛出 ValueError。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)  # 切换到 DAG 模式
        enc1 = builder.add_infer(NT.TEXT_ENCODE)
        enc2 = builder.add_infer(NT.VAE_DECODE)
        gen = builder.add_infer(NT.GENERATE)

        builder.connect(
            (enc1.POSITIVE, gen.POSITIVE),
            (enc2.POSITIVE, gen.POSITIVE),  # 重复输入
        )

        with self.assertRaises(ValueError, msg="Duplicate input"):
            builder.context_builder(_DummyContextBuilder).build()

    def test_format1_uniqueness_raises(self):
        """格式 1 唯一性检测 — 同类型多实例用格式 1 时报错。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_infer(NT.TEXT_ENCODE)
        builder.add_infer(NT.TEXT_ENCODE)  # 同类型两个实例

        with self.assertRaises(ValueError, msg="must be unique"):
            builder.connect(
                (NT.TEXT_ENCODE, TensorKey.POSITIVE, NT.TEXT_ENCODE, TensorKey.POSITIVE),
            )

    def test_duplicate_loader_model_key_raises(self):
        """两个 loader 加载同一 ModelKey 时 build() 报错。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        builder.add_loader(ModelKey.T5TextEncoder)  # 重复

        with self.assertRaises(ValueError, msg="Duplicate loader ModelKey"):
            builder.context_builder(_DummyContextBuilder).build()


# ── PipelineDef 注册表测试 ───────────────────────────────────────────────


class TestPipelineDefRegistry(unittest.TestCase):
    """register_pipeline_def / get_pipeline_def 注册表。"""

    def test_auto_registration_wan_t2v(self):
        """导入 defs 后 Wan2_2_T2V_14B 已注册。"""
        import kdit.pipelines.defs.wan_t2v  # noqa: F401  # pylint: disable=unused-import

        pipeline_def = get_pipeline_def(PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_2_T2V_14B)

    def test_auto_registration_wan_i2v(self):
        """导入 defs 后 Wan2_2_I2V_14B 已注册。"""
        import kdit.pipelines.defs.wan_i2v  # noqa: F401  # pylint: disable=unused-import

        pipeline_def = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_2_I2V_14B)

    def test_auto_registration_wan_vace(self):
        """导入 defs 后 Wan2_1_VACE_14B 已注册。"""
        import kdit.pipelines.defs.wan_vace  # noqa: F401  # pylint: disable=unused-import

        pipeline_def = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_1_VACE_14B)

    def test_auto_registration_qwen_t2i(self):
        """导入 defs 后 QwenImage_T2I 已注册。"""
        import kdit.pipelines.defs.qwen_t2i  # noqa: F401  # pylint: disable=unused-import

        pipeline_def = get_pipeline_def(PipelineKey.QwenImage_T2I)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.QwenImage_T2I)

    def test_auto_registration_qwen_edit(self):
        """导入 defs 后 QwenImage_Edit 已注册。"""
        import kdit.pipelines.defs.qwen_edit  # noqa: F401  # pylint: disable=unused-import

        pipeline_def = get_pipeline_def(PipelineKey.QwenImage_Edit)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.QwenImage_Edit)

    def test_all_defs_registered_via_init(self):
        """通过 __init__.py 导入后所有 5 个 PipelineDef 都已注册。"""
        import kdit.pipelines  # noqa: F401  # pylint: disable=unused-import

        for key in [
            PipelineKey.Wan2_2_T2V_14B,
            PipelineKey.Wan2_2_I2V_14B,
            PipelineKey.Wan2_1_VACE_14B,
            PipelineKey.QwenImage_T2I,
            PipelineKey.QwenImage_Edit,
        ]:
            pipeline_def = get_pipeline_def(key)
            self.assertEqual(pipeline_def.pipeline_key, key)

    def test_get_unregistered_key_raises(self):
        """获取未注册的 key 时抛出 KeyError。"""
        with self.assertRaises(KeyError):
            get_pipeline_def("nonexistent_key")


# ── PipelineDef 结构校验 ────────────────────────────────────────────────


class TestPipelineDefStructure(unittest.TestCase):
    """校验各 PipelineDef 的结构正确性。"""

    @classmethod
    def setUpClass(cls):
        import kdit.pipelines  # noqa: F401  # pylint: disable=unused-import — 触发自动注册

    def test_wan_t2v_structure(self):
        """Wan T2V: 3 load + 5 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)
        self.assertEqual(d.keep_tensors, (TensorKey.VIDEO,))

        # 所有 infer 无条件
        for ip in d.infer_phases:
            self.assertIsNone(ip.condition)

        # node_type 顺序
        types = [ip.node_type for ip in d.infer_phases]
        self.assertEqual(types, [NT.TEXT_ENCODE, NT.VAE_COMPUTE_SHAPE, NT.GENERATE, NT.VAE_DECODE, NT.SAVE_VIDEO])

    def test_wan_i2v_structure(self):
        """Wan I2V: 3 load + 5 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)

        # VAE_ENCODE_SPATIAL 无条件（I2V def 不再使用 .when）
        encode_phase = d.infer_phases[1]  # TEXT_ENCODE, VAE_ENCODE_SPATIAL, ...
        self.assertEqual(encode_phase.node_type, NT.VAE_ENCODE_SPATIAL)
        self.assertIsNone(encode_phase.condition)

    def test_wan_vace_structure(self):
        """Wan VACE: 与 I2V 结构相同，但使用不同的 model_key。"""
        d = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)

        # 检查 model_key 不同于 I2V
        diffusion_phase = next(lp for lp in d.load_phases if lp.model_key == ModelKey.Wan2_1_VACE_14B)
        self.assertEqual(diffusion_phase.model_key, ModelKey.Wan2_1_VACE_14B)

        vae_phase = next(lp for lp in d.load_phases if lp.model_key == ModelKey.VAE_WAN2_1)
        self.assertEqual(vae_phase.model_key, ModelKey.VAE_WAN2_1)

    def test_qwen_t2i_structure(self):
        """Qwen T2I: 3 load + 5 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_T2I)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)

        types = [ip.node_type for ip in d.infer_phases]
        self.assertEqual(types, [NT.TEXT_ENCODE, NT.VAE_COMPUTE_SHAPE, NT.GENERATE, NT.VAE_DECODE, NT.SAVE_IMAGE])

    def test_qwen_edit_structure(self):
        """Qwen Edit: 3 load + 6 infer, VAE_ENCODE_IMAGES 有条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_Edit)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 6)

        encode_phase = d.infer_phases[2]  # TEXT_ENCODE, VAE_COMPUTE_SHAPE, VAE_ENCODE_IMAGES, ...
        self.assertEqual(encode_phase.node_type, NT.VAE_ENCODE_IMAGES)
        self.assertEqual(encode_phase.condition, "has_ref_images")

    def test_wan_i2v_and_vace_share_context_builder(self):
        """Wan I2V 和 VACE 共享同一个 ContextBuilder 类。"""
        d_i2v = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        d_vace = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        self.assertIs(d_i2v.context_builder_cls, d_vace.context_builder_cls)


if __name__ == "__main__":
    unittest.main()
