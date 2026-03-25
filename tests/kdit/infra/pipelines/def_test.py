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

"""PipelineDefBuilder / PipelineDef 注册表 / 结构校验 单元测试。"""

import unittest

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.nodes.core.node_types import IONodeType
from kdit.pipelines.context_builder import ContextBuilder
from kdit.pipelines.pin_ref import NodeRef
from kdit.pipelines.pipeline_def import (
    PipelineDef,
    PipelineDefBuilder,
    _NodeRefWithWhen,
    get_pipeline_def,
)
from kdit.pipelines.pipeline_key import PipelineKey
from kdit.tensor import TensorKey

# ── 测试用 ContextBuilder ────────────────────────────────────────────────


class _DummyContextBuilder(ContextBuilder):
    """最小可用的 ContextBuilder — 用于 Builder 测试。"""

    def build_context(self, node_def, inputs):
        return NodeContext()


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

    def test_add_infer_returns_node_ref_with_when(self):
        """add_infer() 返回 _NodeRefWithWhen（支持 .when() 和 pin 访问）。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        ref = builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)

        self.assertIsInstance(ref, _NodeRefWithWhen)
        self.assertIsInstance(ref, NodeRef)
        self.assertEqual(ref.node_id, 1)

    def test_alloc_node_id_increments(self):
        """_alloc_node_id() 从 0 开始递增。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(builder._alloc_node_id(), 0)
        self.assertEqual(builder._alloc_node_id(), 1)
        self.assertEqual(builder._alloc_node_id(), 2)

    def test_connect_pinref_with_rshift(self):
        """connect() 使用 >> 操作符正确添加 Edge。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        t5 = builder.add_loader(ModelKey.T5TextEncoder)
        enc = builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)

        builder.connect(
            t5.T5TextEncoder >> enc.T5TextEncoder,
        )

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        self.assertEqual(len(pipeline_def.edges), 1)

        edge = pipeline_def.edges[0]
        self.assertEqual(edge.src_node_id, 0)
        self.assertEqual(edge.src_pin, ModelKey.T5TextEncoder)
        self.assertEqual(edge.dst_node_id, 1)
        self.assertEqual(edge.dst_pin, ModelKey.T5TextEncoder)

    def test_connect_pinref_tuple(self):
        """connect() 格式 2（PinRef 元组）正确添加 Edge。"""
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

    def test_connect_one_to_many(self):
        """connect() 一对多 (src, [dst1, dst2]) 生成多条 Edge。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
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
            t5.T5TextEncoder >> enc.T5TextEncoder,
            enc.VIDEO >> save.VIDEO,
        )

        pipeline_def = builder.keep_tensors(TensorKey.VIDEO).context_builder(_DummyContextBuilder).build()

        self.assertIsInstance(pipeline_def, PipelineDef)
        self.assertEqual(pipeline_def.pipeline_key, PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(len(pipeline_def.nodes), 3)
        self.assertEqual(len(pipeline_def.edges), 2)
        self.assertEqual(pipeline_def.keep_tensors, (TensorKey.VIDEO,))
        self.assertIs(pipeline_def.context_builder_cls, _DummyContextBuilder)

        # frozen
        with self.assertRaises(AttributeError):
            pipeline_def.pipeline_key = PipelineKey.QwenImage_T2I

    def test_when_condition(self):
        """add_infer().when() 设置条件。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)
        builder.add_loader(ModelKey.VAE_WAN2_2)
        builder.add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2).when("has_test_input")
        builder.add_infer(NT.SAVE_VIDEO)

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()

        # VAE_ENCODE_SPATIAL 有条件
        vae_node = next(n for n in pipeline_def.nodes if n.node_type == NT.VAE_ENCODE_SPATIAL)
        self.assertEqual(vae_node.condition, "has_test_input")
        # SAVE_VIDEO 无条件
        save_node = next(n for n in pipeline_def.nodes if n.node_type == NT.SAVE_VIDEO)
        self.assertIsNone(save_node.condition)

    def test_when_returns_node_ref(self):
        """add_infer().when() 返回 NodeRef（可用于 connect）。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)
        builder.add_loader(ModelKey.VAE_WAN2_2)
        ref = builder.add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2).when("has_test_input")
        self.assertIsInstance(ref, NodeRef)
        # when() 返回的是普通 NodeRef，不再是 _NodeRefWithWhen
        self.assertNotIsInstance(ref, _NodeRefWithWhen)

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
        self.assertEqual(loader.node_type, IONodeType.LOAD_MODEL)

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

    def test_save_node_no_model_key(self):
        """SaveNode 的 model_key 为 None。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.VAE_WAN2_2)
        builder.add_infer(NT.SAVE_VIDEO)

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        save_node = next(n for n in pipeline_def.nodes if n.node_type == NT.SAVE_VIDEO)
        self.assertIsNone(save_node.model_key)

    def test_frozen_dataclass(self):
        """PipelineDef 是 frozen dataclass — 不可修改。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.VAE_WAN2_2)
        builder.add_infer(NT.SAVE_VIDEO)

        pipeline_def = builder.context_builder(_DummyContextBuilder).build()
        with self.assertRaises(AttributeError):
            pipeline_def.pipeline_key = PipelineKey.QwenImage_T2I

    def test_connect_rejects_4_tuple(self):
        """connect() 不再接受 4-tuple 格式。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        builder.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        builder.add_infer(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)

        with self.assertRaises(ValueError, msg="expects 2-tuple"):
            builder.connect(
                (NT.TEXT_ENCODE, TensorKey.POSITIVE, NT.GENERATE, TensorKey.POSITIVE),
            )

    # ── DAG 校验错误 ──

    def test_type_mismatch_raises(self):
        """TensorKey 连 ModelKey 时 connect() 抛出 TypeError。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        enc = builder.add_infer(NT.TEXT_ENCODE)
        gen = builder.add_infer(NT.GENERATE)

        with self.assertRaises(TypeError, msg="type mismatch"):
            builder.connect(
                enc.POSITIVE >> gen.T5TextEncoder,  # TensorKey → ModelKey
            )

    def test_duplicate_input_raises(self):
        """同一 dst pin 有两条入边时 build() 抛出 ValueError。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_T2V_14B)
        builder.add_loader(ModelKey.T5TextEncoder)
        enc1 = builder.add_infer(NT.TEXT_ENCODE)
        enc2 = builder.add_infer(NT.VAE_DECODE)
        gen = builder.add_infer(NT.GENERATE)

        builder.connect(
            enc1.POSITIVE >> gen.POSITIVE,
            enc2.POSITIVE >> gen.POSITIVE,  # 重复输入
        )

        with self.assertRaises(ValueError, msg="Duplicate input"):
            builder.context_builder(_DummyContextBuilder).build()

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
    """校验各 PipelineDef 的结构正确性（DAG 模式）。"""

    @classmethod
    def setUpClass(cls):
        import kdit.pipelines  # noqa: F401  # pylint: disable=unused-import — 触发自动注册

    def _get_loaders(self, pipeline_def):
        """获取 PipelineDef 中的 loader 节点。"""
        return [n for n in pipeline_def.nodes if n.is_loader]

    def _get_infers(self, pipeline_def):
        """获取 PipelineDef 中的 infer 节点。"""
        return [n for n in pipeline_def.nodes if not n.is_loader]

    def test_wan_t2v_structure(self):
        """Wan T2V: 3 loader + 5 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_T2V_14B)
        loaders = self._get_loaders(d)
        infers = self._get_infers(d)

        self.assertEqual(len(loaders), 3)
        self.assertEqual(len(infers), 5)
        self.assertEqual(d.keep_tensors, (TensorKey.VIDEO,))

        # 所有 infer 无条件
        for nd in infers:
            self.assertIsNone(nd.condition)

        # node_type 顺序
        types = [nd.node_type for nd in infers]
        self.assertEqual(
            types,
            [
                NT.TEXT_ENCODE,
                NT.VAE_COMPUTE_SHAPE,
                NT.GENERATE,
                NT.VAE_DECODE,
                NT.SAVE_VIDEO,
            ],
        )

    def test_wan_i2v_structure(self):
        """Wan I2V: 3 loader + 多个 infer（含 READ_IMAGE 等）。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        loaders = self._get_loaders(d)
        infers = self._get_infers(d)

        self.assertEqual(len(loaders), 3)
        self.assertGreaterEqual(len(infers), 5)

        # 检查包含 READ_IMAGE 节点
        read_image_nodes = [n for n in infers if n.node_type == NT.READ_IMAGE]
        self.assertGreaterEqual(len(read_image_nodes), 1)

        # 检查 VACE_PREPROCESS 有条件
        vace_nodes = [n for n in infers if n.node_type == NT.VACE_PREPROCESS]
        if vace_nodes:
            self.assertEqual(vace_nodes[0].condition, "has_vace")

    def test_wan_vace_structure(self):
        """Wan VACE: 与 I2V 结构相同，但使用不同的 model_key。"""
        d = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        loaders = self._get_loaders(d)
        infers = self._get_infers(d)

        self.assertEqual(len(loaders), 3)
        self.assertGreaterEqual(len(infers), 5)

        # 检查 model_key 不同于 I2V
        loader_keys = {n.model_key for n in loaders}
        self.assertIn(ModelKey.Wan2_1_VACE_14B, loader_keys)
        self.assertIn(ModelKey.VAE_WAN2_1, loader_keys)

    def test_qwen_t2i_structure(self):
        """Qwen T2I: 3 loader + 5 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_T2I)
        loaders = self._get_loaders(d)
        infers = self._get_infers(d)

        self.assertEqual(len(loaders), 3)
        self.assertEqual(len(infers), 5)

        types = [nd.node_type for nd in infers]
        self.assertEqual(
            types,
            [
                NT.TEXT_ENCODE,
                NT.VAE_COMPUTE_SHAPE,
                NT.GENERATE,
                NT.VAE_DECODE,
                NT.SAVE_IMAGE,
            ],
        )

    def test_qwen_edit_structure(self):
        """Qwen Edit: 3 loader + 多个 infer, VAE_ENCODE_IMAGES 有条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_Edit)
        loaders = self._get_loaders(d)
        infers = self._get_infers(d)

        self.assertEqual(len(loaders), 3)
        self.assertGreaterEqual(len(infers), 5)

        # 检查 VAE_ENCODE_IMAGES 有条件
        encode_nodes = [n for n in infers if n.node_type == NT.VAE_ENCODE_IMAGES]
        self.assertEqual(len(encode_nodes), 1)
        self.assertEqual(encode_nodes[0].condition, "has_ref_images")

    def test_wan_i2v_and_vace_share_context_builder(self):
        """Wan I2V 和 VACE 共享同一个 ContextBuilder 类。"""
        d_i2v = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        d_vace = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        self.assertIs(d_i2v.context_builder_cls, d_vace.context_builder_cls)

    def test_all_defs_have_edges(self):
        """所有 PipelineDef 都有 edges（DAG 模式）。"""
        for key in [
            PipelineKey.Wan2_2_T2V_14B,
            PipelineKey.Wan2_2_I2V_14B,
            PipelineKey.Wan2_1_VACE_14B,
            PipelineKey.QwenImage_T2I,
            PipelineKey.QwenImage_Edit,
        ]:
            d = get_pipeline_def(key)
            self.assertGreater(len(d.edges), 0, f"{key} should have edges")
            self.assertGreater(len(d.nodes), 0, f"{key} should have nodes")


if __name__ == "__main__":
    unittest.main()
