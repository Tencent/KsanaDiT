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


# ── PipelineDefBuilder 测试 ──────────────────────────────────────────────


class TestPipelineDefBuilder(unittest.TestCase):
    """PipelineDefBuilder 链式构建与校验。"""

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
            .when("has_start_image")
            .add_infer(NT.SAVE_VIDEO)
            .context_builder(_DummyContextBuilder)
            .build()
        )

        # VAE_ENCODE_SPATIAL 有条件
        self.assertEqual(pipeline_def.infer_phases[0].condition, "has_start_image")
        # SAVE_VIDEO 无条件
        self.assertIsNone(pipeline_def.infer_phases[1].condition)

    def test_when_chain_returns_builder(self):
        """add_infer().when() 返回 PipelineDefBuilder，可继续链式。"""
        builder = PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)
        builder.load(ModelKey.VAE_WAN2_2)
        chain = builder.add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2)
        self.assertIsInstance(chain, _InferTaskChain)
        result = chain.when("has_start_image")
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
