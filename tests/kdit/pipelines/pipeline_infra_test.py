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

"""Pipeline V4 声明式架构的纯逻辑单元测试。

不需要 GPU / 模型 / 网络，只测试：
- PipelineDefBuilder 构建与校验
- PipelineDef 注册表
- 所有 defs 的自动注册
- ContextBuilder 基类逻辑
- Wan / Qwen ContextBuilder 的 prepare / build / condition
- 辅助函数
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kdit.config.sample_config import SampleConfig
from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.context_builder import ContextBuilder
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline import (
    _ensure_cache_config_list,
    _get_num_prompts,
    _merge_sample_config,
)
from kdit.pipelines.pipeline_def import (
    InferPhase,
    PipelineDef,
    PipelineDefBuilder,
    _InferPhaseChain,
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
        self.assertIsInstance(chain, _InferPhaseChain)
        result = chain.when("has_start_image")
        self.assertIs(result, builder)

    def test_chain_proxy_without_when(self):
        """不调用 .when() 时，_InferPhaseChain 代理到 builder。"""
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
        with self.assertRaises(ValueError, msg="not declared in any LoadPhase"):
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
        """Wan T2V: 3 load + 4 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_T2V_14B)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 4)
        self.assertEqual(d.keep_tensors, (TensorKey.VIDEO,))

        # 所有 infer 无条件
        for ip in d.infer_phases:
            self.assertIsNone(ip.condition)

        # node_type 顺序
        types = [ip.node_type for ip in d.infer_phases]
        self.assertEqual(types, [NT.TEXT_ENCODE, NT.GENERATE, NT.VAE_DECODE, NT.SAVE_VIDEO])

    def test_wan_i2v_structure(self):
        """Wan I2V: 3 load + 5 infer, VAE_ENCODE_SPATIAL 有条件。"""
        d = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)

        # VAE_ENCODE_SPATIAL 有条件
        encode_phase = d.infer_phases[1]  # TEXT_ENCODE, VAE_ENCODE_SPATIAL, ...
        self.assertEqual(encode_phase.node_type, NT.VAE_ENCODE_SPATIAL)
        self.assertEqual(encode_phase.condition, "has_start_image")

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
        """Qwen T2I: 3 load + 4 infer, 无条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_T2I)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 4)

        types = [ip.node_type for ip in d.infer_phases]
        self.assertEqual(types, [NT.TEXT_ENCODE, NT.GENERATE, NT.VAE_DECODE, NT.SAVE_IMAGE])

    def test_qwen_edit_structure(self):
        """Qwen Edit: 3 load + 5 infer, VAE_ENCODE_IMAGES 有条件。"""
        d = get_pipeline_def(PipelineKey.QwenImage_Edit)
        self.assertEqual(len(d.load_phases), 3)
        self.assertEqual(len(d.infer_phases), 5)

        encode_phase = d.infer_phases[1]
        self.assertEqual(encode_phase.node_type, NT.VAE_ENCODE_IMAGES)
        self.assertEqual(encode_phase.condition, "has_ref_images")

    def test_wan_i2v_and_vace_share_context_builder(self):
        """Wan I2V 和 VACE 共享同一个 ContextBuilder 类。"""
        d_i2v = get_pipeline_def(PipelineKey.Wan2_2_I2V_14B)
        d_vace = get_pipeline_def(PipelineKey.Wan2_1_VACE_14B)
        self.assertIs(d_i2v.context_builder_cls, d_vace.context_builder_cls)


# ── ContextBuilder 基类测试 ──────────────────────────────────────────────


class TestContextBuilderBase(unittest.TestCase):
    """ContextBuilder 基类的 check_condition 和 post_process。"""

    def _make_inputs(self, **overrides) -> PipelineGenerateInputs:
        """创建最小 GenerateInputs。"""
        defaults = {
            "prompt": "test",
            "prompt_negative": None,
            "num_prompts": 1,
            "sample_config": MagicMock(),
            "runtime_config": MagicMock(offload_model=False, save_output=False),
            "cache_config": None,
            "has_lora": False,
        }
        defaults.update(overrides)
        return PipelineGenerateInputs(**defaults)

    def test_check_condition_calls_method(self):
        """check_condition 调用 self 上的同名方法。"""
        builder = _DummyContextBuilder()
        builder.my_condition = MagicMock(return_value=True)
        inputs = self._make_inputs()
        result = builder.check_condition("my_condition", inputs)
        self.assertTrue(result)
        builder.my_condition.assert_called_once_with(inputs)

    def test_check_condition_missing_raises(self):
        """check_condition 找不到方法时抛出 ValueError。"""
        builder = _DummyContextBuilder()
        inputs = self._make_inputs()
        with self.assertRaises(ValueError, msg="Condition 'nonexistent' not found"):
            builder.check_condition("nonexistent", inputs)

    def test_prepare_tensors_default_none(self):
        """默认 prepare_tensors 返回 None。"""
        builder = _DummyContextBuilder()
        result = builder.prepare_tensors(MagicMock(), self._make_inputs())
        self.assertIsNone(result)

    def test_post_process_default_passthrough(self):
        """默认 post_process 直接返回输入。"""
        builder = _DummyContextBuilder()
        sentinel = object()
        result = builder.post_process(sentinel, self._make_inputs())
        self.assertIs(result, sentinel)

    def test_common_metadata(self):
        """_common_metadata 包含 offload_model 和 text_run_device。"""
        inputs = self._make_inputs()
        inputs.runtime_config.offload_model = True
        meta = ContextBuilder._common_metadata(inputs)
        self.assertTrue(meta["offload_model"])
        self.assertIn("text_run_device", meta)


# ── Wan ContextBuilder 测试 ──────────────────────────────────────────────


def _make_wan_settings():
    """创建模拟的 default_settings（Wan 系列）。"""
    return SimpleNamespace(
        vae=SimpleNamespace(z_dim=16, stride=[4, 8, 8]),
        diffusion=SimpleNamespace(patch_size=[1, 2, 2]),
        sample_config=MagicMock(),
        runtime_config=MagicMock(),
    )


def _make_inputs(prompt="test", num_prompts=1, **overrides) -> PipelineGenerateInputs:
    """创建最小 GenerateInputs。"""
    rc = MagicMock()
    rc.size = (720, 480)
    rc.frame_num = 17
    rc.offload_model = False
    rc.save_output = False
    rc.return_frames = True
    rc.output_folder = "outputs"
    rc.batch_size_per_prompts = [1] * num_prompts
    defaults = {
        "prompt": prompt,
        "prompt_negative": None,
        "num_prompts": num_prompts,
        "sample_config": MagicMock(fps=30),
        "runtime_config": rc,
        "cache_config": None,
        "has_lora": False,
    }
    defaults.update(overrides)
    return PipelineGenerateInputs(**defaults)


class TestWanT2VContextBuilder(unittest.TestCase):
    """WanT2VContextBuilder 的 prepare / build。"""

    def test_prepare_computes_noise_shape(self):
        """prepare_generate_inputs 计算 noise_shape。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNotNone(builder._extra)
        self.assertIsInstance(builder._extra.noise_shape, list)
        self.assertEqual(len(builder._extra.noise_shape), 4)

    def test_prepare_missing_settings_raises(self):
        """缺少 _default_settings 时抛出 ValueError。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        with self.assertRaises(ValueError, msg="_default_settings"):
            builder.prepare_generate_inputs(inputs)

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs(prompt="hello world")
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        phase = InferPhase(node_type=NT.TEXT_ENCODE, model_key=ModelKey.T5TextEncoder)
        ctx = builder.build_context(phase, inputs)

        self.assertIsInstance(ctx, NodeContext)
        self.assertEqual(ctx.prompt, "hello world")

    def test_build_context_generate(self):
        """build_context(GENERATE) 返回包含 noise_shape 的 context。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        phase = InferPhase(node_type=NT.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)
        ctx = builder.build_context(phase, inputs)

        self.assertIn("noise_shape", ctx.metadata)
        self.assertIsNotNone(ctx.metadata["noise_shape"])

    def test_build_context_unexpected_type_raises(self):
        """build_context 遇到未知 node_type 时抛出 ValueError。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        phase = InferPhase(node_type=NT.VAE_ENCODE_IMAGES)
        with self.assertRaises(ValueError, msg="unexpected node_type"):
            builder.build_context(phase, inputs)


class TestWanI2VContextBuilder(unittest.TestCase):
    """WanI2VContextBuilder 的 prepare / build / condition / prepare_tensors。"""

    def test_prepare_with_no_image(self):
        """无图时 noise_shape 不为 None，start_img_path 为 None。"""
        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNone(builder._extra.start_img_path)
        self.assertIsNotNone(builder._extra.noise_shape)
        self.assertFalse(builder._extra.with_end_image)

    def test_has_start_image_false_when_no_image(self):
        """无图时 has_start_image 返回 False。"""
        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        self.assertFalse(builder.has_start_image(inputs))

    @patch("kdit.pipelines.context_builders.wan._load_image")
    def test_prepare_with_image(self, mock_load):
        """有图时 noise_shape 为 None，start_img_path 不为 None。"""
        import torch

        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        mock_load.return_value = torch.zeros(1, 3, 480, 720)

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()

        builder.prepare_generate_inputs(
            inputs,
            _default_settings=settings,
            start_img_path="test.png",
        )

        self.assertIsNotNone(builder._extra.start_img_path)
        self.assertIsNone(builder._extra.noise_shape)
        self.assertTrue(builder.has_start_image(inputs))

    @patch("kdit.pipelines.context_builders.wan._load_image")
    def test_prepare_with_end_image(self, mock_load):
        """有 end_img 时 with_end_image 为 True。"""
        import torch

        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        mock_load.return_value = torch.zeros(1, 3, 480, 720)

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()

        builder.prepare_generate_inputs(
            inputs,
            _default_settings=settings,
            start_img_path="start.png",
            end_img_path="end.png",
        )

        self.assertTrue(builder._extra.with_end_image)

    @patch("kdit.pipelines.context_builders.wan._load_image")
    def test_prepare_tensors_vae_encode(self, mock_load):
        """prepare_tensors(VAE_ENCODE_SPATIAL) 返回 START_IMG 和 END_IMG。"""
        import torch

        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        mock_load.return_value = torch.zeros(1, 3, 480, 720)

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(
            inputs,
            _default_settings=_make_wan_settings(),
            start_img_path="test.png",
        )

        phase = InferPhase(node_type=NT.VAE_ENCODE_SPATIAL, model_key=ModelKey.VAE_WAN2_2)
        tensors = builder.prepare_tensors(phase, inputs)

        self.assertIsNotNone(tensors)
        self.assertIn(TensorKey.START_IMG, tensors)

    def test_prepare_tensors_generate_no_latent(self):
        """无 input_latent 时 prepare_tensors(GENERATE) 返回 None。"""
        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        phase = InferPhase(node_type=NT.GENERATE, model_key=ModelKey.Wan2_2_I2V_14B)
        tensors = builder.prepare_tensors(phase, inputs)
        self.assertIsNone(tensors)

    def test_build_context_vae_encode_spatial(self):
        """build_context(VAE_ENCODE_SPATIAL) 返回包含 target_f/h/w 的 context。"""
        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(inputs, _default_settings=_make_wan_settings())

        phase = InferPhase(node_type=NT.VAE_ENCODE_SPATIAL, model_key=ModelKey.VAE_WAN2_2)
        ctx = builder.build_context(phase, inputs)

        self.assertIn("target_f", ctx.metadata)
        self.assertIn("target_h", ctx.metadata)
        self.assertIn("target_w", ctx.metadata)


# ── Qwen ContextBuilder 测试 ─────────────────────────────────────────────


def _make_qwen_settings():
    """创建模拟的 default_settings（Qwen 系列）。"""
    return SimpleNamespace(
        vae=SimpleNamespace(z_dim=16, vae_scale_factor=8),
        diffusion=SimpleNamespace(patch_size=2),
        sample_config=MagicMock(),
        runtime_config=MagicMock(),
    )


class TestQwenT2IContextBuilder(unittest.TestCase):
    """QwenT2IContextBuilder 的 prepare / build。"""

    def test_prepare_computes_noise_shape(self):
        """prepare_generate_inputs 计算 noise_shape。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNotNone(builder._extra)
        self.assertIsInstance(builder._extra.noise_shape, list)
        self.assertEqual(len(builder._extra.noise_shape), 4)

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs(prompt="a cat")
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(inputs, _default_settings=_make_qwen_settings())

        phase = InferPhase(node_type=NT.TEXT_ENCODE, model_key=ModelKey.Qwen2VLTextEncoder)
        ctx = builder.build_context(phase, inputs)

        self.assertEqual(ctx.prompt, "a cat")
        # T2I 不传 condition_image_path
        self.assertNotIn("condition_image_path", ctx.metadata)

    def test_build_context_save_image(self):
        """build_context(SAVE_IMAGE) 返回 context。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(inputs, _default_settings=_make_qwen_settings())

        phase = InferPhase(node_type=NT.SAVE_IMAGE)
        ctx = builder.build_context(phase, inputs)
        self.assertIsInstance(ctx, NodeContext)


class TestQwenEditContextBuilder(unittest.TestCase):
    """QwenEditContextBuilder 的 prepare / build / condition / prepare_tensors。"""

    def test_prepare_with_no_image(self):
        """无参考图时 img_path 为 None。"""
        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNone(builder._extra.img_path)
        self.assertIsNotNone(builder._extra.noise_shape)

    def test_has_ref_images_false_when_no_image(self):
        """无参考图时 has_ref_images 返回 False。"""
        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(inputs, _default_settings=_make_qwen_settings())

        self.assertFalse(builder.has_ref_images(inputs))

    @patch("kdit.pipelines.context_builders.qwen._load_ref_images")
    def test_prepare_with_ref_images(self, mock_load):
        """有参考图时 img_path 不为 None。"""
        import torch

        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        mock_load.return_value = [torch.zeros(1, 3, 1024, 1024)]

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)

        builder.prepare_generate_inputs(
            inputs,
            _default_settings=_make_qwen_settings(),
            img_path="ref.png",
        )

        self.assertIsNotNone(builder._extra.img_path)
        self.assertTrue(builder.has_ref_images(inputs))

    @patch("kdit.pipelines.context_builders.qwen._load_ref_images")
    def test_build_context_text_encode_with_condition_image(self, mock_load):
        """Edit 模式 build_context(TEXT_ENCODE) 包含 condition_image_path。"""
        import torch

        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        mock_load.return_value = [torch.zeros(1, 3, 1024, 1024)]

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(
            inputs,
            _default_settings=_make_qwen_settings(),
            img_path="ref.png",
        )

        phase = InferPhase(node_type=NT.TEXT_ENCODE, model_key=ModelKey.Qwen2VLTextEncoderMultimodal)
        ctx = builder.build_context(phase, inputs)

        self.assertIn("condition_image_path", ctx.metadata)

    @patch("kdit.pipelines.context_builders.qwen._load_ref_images")
    def test_prepare_tensors_vae_encode_images(self, mock_load):
        """prepare_tensors(VAE_ENCODE_IMAGES) 返回 IMAGE tensor。"""
        import torch

        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        mock_load.return_value = [torch.zeros(1, 3, 1024, 1024)]

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(
            inputs,
            _default_settings=_make_qwen_settings(),
            img_path="ref.png",
        )

        phase = InferPhase(node_type=NT.VAE_ENCODE_IMAGES, model_key=ModelKey.QwenImageVAE)
        tensors = builder.prepare_tensors(phase, inputs)

        self.assertIsNotNone(tensors)
        self.assertIn(TensorKey.IMAGE, tensors)


# ── 辅助函数测试 ─────────────────────────────────────────────────────────


class TestHelperFunctions(unittest.TestCase):
    """Pipeline 模块级辅助函数。"""

    def test_get_num_prompts_str(self):
        self.assertEqual(_get_num_prompts("hello"), 1)

    def test_get_num_prompts_list(self):
        self.assertEqual(_get_num_prompts(["a", "b"]), 2)

    def test_get_num_prompts_empty_list(self):
        self.assertEqual(_get_num_prompts([]), 0)

    def test_get_num_prompts_invalid(self):
        self.assertEqual(_get_num_prompts(123), 0)

    def test_merge_sample_config_none_returns_sample_config(self):
        default = MagicMock()
        default.steps = None
        default.shift = None
        default.denoise = None
        default.cfg_scale = None
        default.solver = None
        result = _merge_sample_config(None, default)
        self.assertIsInstance(result, SampleConfig)

    def test_ensure_cache_config_list_none_returns_none(self):
        default = MagicMock()
        result = _ensure_cache_config_list(None, default)
        self.assertIsNone(result)

    def test_ensure_cache_config_list_passthrough(self):
        from kdit.cache import CacheConfig

        config = CacheConfig()
        result = _ensure_cache_config_list(config, MagicMock())
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], config)

    def test_resolve_lora_config_single(self):
        from kdit.config.lora_config import LoraConfig
        from kdit.pipelines.context_builder import ContextBuilder

        class _TestBuilder(ContextBuilder):
            def build_context(self, phase, inputs):
                return None

        builder = _TestBuilder()
        lora = LoraConfig("/path/to/lora")
        result = builder.resolve_lora_config(lora, MagicMock())
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)


class TestWanHelperFunctions(unittest.TestCase):
    """Wan context_builders 的辅助函数。"""

    def test_valid_images_none(self):
        from kdit.pipelines.context_builders.wan import _valid_images

        result = _valid_images(None, 2)
        self.assertIsNone(result)

    def test_valid_images_single_string(self):
        from kdit.pipelines.context_builders.wan import _valid_images

        result = _valid_images("test.png", 3)
        self.assertEqual(result, ["test.png"])

    def test_valid_images_matching_list(self):
        from kdit.pipelines.context_builders.wan import _valid_images

        result = _valid_images(["a.png", "b.png"], 2)
        self.assertEqual(result, ["a.png", "b.png"])

    def test_valid_images_mismatched_raises(self):
        from kdit.pipelines.context_builders.wan import _valid_images

        with self.assertRaises(ValueError):
            _valid_images(["a.png", "b.png"], 3)

    def test_compute_save_path_no_save(self):
        from kdit.pipelines.context_builders import compute_save_path

        inputs = _make_inputs()
        inputs.runtime_config.save_output = False
        result = compute_save_path(inputs, prefix="wan", ext=".mp4")
        self.assertIsNone(result)

    def test_compute_save_path_with_save(self):
        from kdit.pipelines.context_builders import compute_save_path

        inputs = _make_inputs()
        inputs.runtime_config.save_output = True
        inputs.runtime_config.output_folder = "/tmp/test"
        result = compute_save_path(inputs, prefix="wan", ext=".mp4")
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("/tmp/test/"))
        self.assertTrue(result.endswith(".mp4"))


class TestQwenHelperFunctions(unittest.TestCase):
    """Qwen context_builders 的辅助函数。"""

    def test_valid_ref_images_none(self):
        from kdit.pipelines.context_builders.qwen import _valid_ref_images

        result = _valid_ref_images(None, 2)
        self.assertIsNone(result)

    def test_valid_ref_images_single_string(self):
        from kdit.pipelines.context_builders.qwen import _valid_ref_images

        result = _valid_ref_images("test.png", 1)
        self.assertEqual(result, [["test.png"]])

    def test_valid_ref_images_flat_list(self):
        from kdit.pipelines.context_builders.qwen import _valid_ref_images

        result = _valid_ref_images(["a.png", "b.png"], 1)
        self.assertEqual(result, [["a.png", "b.png"]])

    def test_valid_ref_images_nested_list(self):
        from kdit.pipelines.context_builders.qwen import _valid_ref_images

        result = _valid_ref_images([["a.png"], ["b.png"]], 2)
        self.assertEqual(result, [["a.png"], ["b.png"]])

    def test_valid_ref_images_mismatched_raises(self):
        from kdit.pipelines.context_builders.qwen import _valid_ref_images

        with self.assertRaises(ValueError):
            _valid_ref_images([["a.png"], ["b.png"]], 3)

    def test_compute_save_path_png(self):
        from kdit.pipelines.context_builders import compute_save_path

        inputs = _make_inputs()
        inputs.runtime_config.save_output = True
        inputs.runtime_config.output_folder = "/tmp/test"
        result = compute_save_path(inputs, prefix="qwen", ext=".png")
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".png"))


# ── __init__.py 导出测试 ─────────────────────────────────────────────────


class TestPipelineExports(unittest.TestCase):
    """kdit.pipelines.__init__.py 导出的公共 API。"""

    def test_pipeline_importable(self):
        from kdit.pipelines import Pipeline

        self.assertTrue(callable(Pipeline))

    def test_pipeline_def_importable(self):
        from kdit.pipelines import PipelineDefBuilder

        self.assertTrue(callable(PipelineDefBuilder))

    def test_context_builder_importable(self):
        from kdit.pipelines import ContextBuilder

        self.assertTrue(callable(ContextBuilder))

    def test_generate_inputs_importable(self):
        from kdit.pipelines import PipelineGenerateInputs

        self.assertTrue(callable(PipelineGenerateInputs))

    def test_pipeline_key_importable(self):
        from kdit.pipelines import PipelineKey

        self.assertTrue(hasattr(PipelineKey, "Wan2_2_T2V_14B"))

    def test_register_functions_importable(self):
        from kdit.pipelines import get_pipeline_def, register_pipeline_def

        self.assertTrue(callable(register_pipeline_def))
        self.assertTrue(callable(get_pipeline_def))

    def test_all_exports(self):
        import kdit.pipelines

        for name in kdit.pipelines.__all__:
            self.assertTrue(hasattr(kdit.pipelines, name), f"{name} not in kdit.pipelines")


if __name__ == "__main__":
    unittest.main()
