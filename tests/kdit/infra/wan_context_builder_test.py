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
from unittest.mock import MagicMock, patch

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline_def import InferPhase
from kdit.tensor import TensorKey


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


# ── WanT2VContextBuilder ─────────────────────────────────────────────────


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


# ── WanI2VContextBuilder ─────────────────────────────────────────────────


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
