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

"""Wan ContextBuilder (T2V / I2V) 及辅助函数单元测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.nodes.core.node_types import IONodeType as IOT
from kdit.pipelines.context_builders.wan import WanI2VExtraInputs
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline_def import NodeDef


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


def _node_def(node_type, model_key=None):
    """创建 InferNode 的 NodeDef。"""
    return NodeDef(node_type=node_type, model_key=model_key)


# ── WanT2VContextBuilder ─────────────────────────────────────────────────


class TestWanT2VContextBuilder(unittest.TestCase):
    """WanT2VContextBuilder 的 prepare / build。"""

    def test_prepare_stores_target_dimensions(self):
        """prepare_generate_inputs 保存目标尺寸。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()

        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=settings,
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        self.assertIsNotNone(builder._extra)
        self.assertEqual(builder._extra.target_f, 17)
        self.assertEqual(builder._extra.target_h, 480)
        self.assertEqual(builder._extra.target_w, 720)

    def test_prepare_missing_settings_raises(self):
        """缺少 _default_settings 时抛出 TypeError。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        with self.assertRaises(TypeError):
            builder.prepare_generate_inputs(inputs, None)

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs(prompt="hello world")
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_wan_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        ctx = builder.build_context(nd, inputs)

        self.assertIsInstance(ctx, NodeContext)
        self.assertEqual(ctx.prompt, "hello world")

    def test_build_context_generate(self):
        """build_context(GENERATE) 返回包含 sample_config 的 context。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_wan_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.GENERATE, ModelKey.Wan2_2_T2V_14B)
        ctx = builder.build_context(nd, inputs)

        self.assertIsNotNone(ctx.sample_config)
        self.assertIsNotNone(ctx.runtime_config)

    def test_build_context_vae_compute_shape(self):
        """build_context(VAE_COMPUTE_SHAPE) 返回包含 target_f/h/w 的 context。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_wan_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.VAE_COMPUTE_SHAPE, ModelKey.VAE_WAN2_2)
        ctx = builder.build_context(nd, inputs)

        self.assertIn("target_f", ctx.metadata)
        self.assertIn("target_h", ctx.metadata)
        self.assertIn("target_w", ctx.metadata)

    def test_build_context_unexpected_type_raises(self):
        """build_context 遇到未知 node_type 时抛出 ValueError。"""
        from kdit.pipelines.context_builders.wan import WanT2VContextBuilder

        builder = WanT2VContextBuilder()
        inputs = _make_inputs()
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_wan_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.VAE_ENCODE_IMAGES)
        with self.assertRaises(ValueError, msg="unexpected node_type"):
            builder.build_context(nd, inputs)


# ── WanI2VContextBuilder ─────────────────────────────────────────────────


class TestWanI2VContextBuilder(unittest.TestCase):
    """WanI2VContextBuilder 的 prepare / build / condition。"""

    def _prepare(self, extra_inputs=None):
        """辅助方法：创建 builder 并调用 prepare_generate_inputs。"""
        from kdit.pipelines.context_builders.wan import WanI2VContextBuilder

        builder = WanI2VContextBuilder()
        inputs = _make_inputs()
        settings = _make_wan_settings()
        builder.prepare_generate_inputs(
            inputs,
            extra_inputs,
            _default_settings=settings,
            _engine=MagicMock(),
            _vae_model_key=ModelKey.VAE_WAN2_2,
        )
        return builder, inputs

    def test_prepare_with_no_image(self):
        """无图时 start_img_path 为 None，target_frame_num 有值。"""
        builder, _ = self._prepare()

        self.assertIsNone(builder._extra.start_img_path)
        self.assertIsNotNone(builder._extra.target_frame_num)
        self.assertFalse(builder._extra.with_end_image)

    def test_prepare_with_start_image(self):
        """有 start_img 时 start_img_path 不为 None。"""
        extra = WanI2VExtraInputs(start_img_path="test.png")
        builder, _ = self._prepare(extra_inputs=extra)

        self.assertIsNotNone(builder._extra.start_img_path)
        self.assertEqual(builder._extra.start_img_path, ["test.png"])

    def test_prepare_with_end_image(self):
        """有 end_img 时 with_end_image 为 True。"""
        extra = WanI2VExtraInputs(start_img_path="start.png", end_img_path="end.png")
        builder, _ = self._prepare(extra_inputs=extra)

        self.assertTrue(builder._extra.with_end_image)

    def test_condition_has_start_image_true(self):
        """有 start_img 时 has_start_image 返回 True。"""
        extra = WanI2VExtraInputs(start_img_path="test.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        self.assertTrue(builder.has_start_image(inputs))

    def test_condition_has_start_image_false(self):
        """无 start_img 时 has_start_image 返回 False。"""
        builder, inputs = self._prepare()

        self.assertFalse(builder.has_start_image(inputs))

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        builder, inputs = self._prepare()

        nd = _node_def(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
        ctx = builder.build_context(nd, inputs)

        self.assertIsInstance(ctx, NodeContext)
        self.assertEqual(ctx.prompt, "test")

    def test_build_context_read_image(self):
        """build_context(READ_IMAGE) 返回包含 img_paths 的 context。"""
        extra = WanI2VExtraInputs(start_img_path="test.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        nd = _node_def(IOT.READ_IMAGE)
        ctx = builder.build_context(nd, inputs)

        self.assertIn("img_paths", ctx.metadata)

    def test_build_context_vae_encode_spatial(self):
        """build_context(VAE_ENCODE_SPATIAL) 返回包含 target_f/h/w 的 context。"""
        builder, inputs = self._prepare()

        nd = _node_def(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_2)
        ctx = builder.build_context(nd, inputs)

        self.assertIn("target_f", ctx.metadata)
        self.assertIn("target_h", ctx.metadata)
        self.assertIn("target_w", ctx.metadata)

    def test_build_context_unexpected_type_raises(self):
        """build_context 遇到未知 node_type 时抛出 ValueError。"""
        builder, inputs = self._prepare()

        nd = _node_def(NT.VAE_ENCODE_IMAGES)
        with self.assertRaises(ValueError, msg="unexpected node_type"):
            builder.build_context(nd, inputs)


# ── Wan 辅助函数 ─────────────────────────────────────────────────────────


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


if __name__ == "__main__":
    unittest.main()
