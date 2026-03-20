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

"""Qwen ContextBuilder (T2I / Edit) 及辅助函数单元测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline_def import InferTask
from kdit.tensor import TensorKey


def _make_qwen_settings():
    """创建模拟的 default_settings（Qwen 系列）。"""
    return SimpleNamespace(
        vae=SimpleNamespace(z_dim=16, stride=[1, 8, 8]),
        diffusion=SimpleNamespace(patch_size=2),
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


# ── QwenT2IContextBuilder ────────────────────────────────────────────────


class TestQwenT2IContextBuilder(unittest.TestCase):
    """QwenT2IContextBuilder 的 prepare / build。"""

    def test_prepare_stores_target_dimensions(self):
        """prepare_generate_inputs 保存目标尺寸（noise_shape 由 VAE_COMPUTE_SHAPE 节点计算）。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNotNone(builder._extra)
        self.assertEqual(builder._extra.target_h, 1024)
        self.assertEqual(builder._extra.target_w, 1024)

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs(prompt="a cat")
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(inputs, _default_settings=_make_qwen_settings())

        phase = InferTask(node_type=NT.TEXT_ENCODE, model_key=ModelKey.Qwen2VLTextEncoder)
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

        phase = InferTask(node_type=NT.SAVE_IMAGE)
        ctx = builder.build_context(phase, inputs)
        self.assertIsInstance(ctx, NodeContext)


# ── QwenEditContextBuilder ───────────────────────────────────────────────


class TestQwenEditContextBuilder(unittest.TestCase):
    """QwenEditContextBuilder 的 prepare / build / condition / prepare_tensors。"""

    def test_prepare_with_no_image(self):
        """无参考图时 img_path 为 None，target_h/w 有值。"""
        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()

        builder.prepare_generate_inputs(inputs, _default_settings=settings)

        self.assertIsNone(builder._extra.img_path)
        self.assertEqual(builder._extra.target_h, 1024)
        self.assertEqual(builder._extra.target_w, 1024)

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

        phase = InferTask(node_type=NT.TEXT_ENCODE, model_key=ModelKey.Qwen2VLTextEncoderMultimodal)
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

        phase = InferTask(node_type=NT.VAE_ENCODE_IMAGES, model_key=ModelKey.QwenImageVAE)
        tensors = builder.prepare_tensors(phase, inputs)

        self.assertIsNotNone(tensors)
        self.assertIn(TensorKey.IMAGE, tensors)


# ── Qwen 辅助函数 ────────────────────────────────────────────────────────


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


if __name__ == "__main__":
    unittest.main()
