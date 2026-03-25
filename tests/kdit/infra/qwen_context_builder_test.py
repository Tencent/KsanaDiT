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
from unittest.mock import MagicMock

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.pipelines.context_builders.qwen import QwenEditExtraInputs
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline_def import NodeDef


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


def _node_def(node_type, model_key=None, node_id=1):
    """创建 InferNode 的 NodeDef。"""
    return NodeDef(node_id=node_id, node_type=node_type, model_key=model_key)


# ── QwenT2IContextBuilder ────────────────────────────────────────────────


class TestQwenT2IContextBuilder(unittest.TestCase):
    """QwenT2IContextBuilder 的 prepare / build。"""

    def test_prepare_stores_target_dimensions(self):
        """prepare_generate_inputs 保存目标尺寸。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()

        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=settings,
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        self.assertIsNotNone(builder._extra)
        self.assertEqual(builder._extra.target_h, 1024)
        self.assertEqual(builder._extra.target_w, 1024)

    def test_build_context_text_encode(self):
        """build_context(TEXT_ENCODE) 返回包含 prompt 的 context。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs(prompt="a cat")
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_qwen_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.TEXT_ENCODE, ModelKey.Qwen2VLTextEncoder)
        ctx = builder.build_context(nd, inputs)

        self.assertEqual(ctx.prompt, "a cat")
        # T2I 不传 condition_image_path
        self.assertNotIn("condition_image_path", ctx.metadata)

    def test_build_context_save_image(self):
        """build_context(SAVE_IMAGE) 返回 context。"""
        from kdit.pipelines.context_builders.qwen import QwenT2IContextBuilder

        builder = QwenT2IContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        builder.prepare_generate_inputs(
            inputs,
            None,
            _default_settings=_make_qwen_settings(),
            _engine=MagicMock(),
            _vae_model_key=None,
        )

        nd = _node_def(NT.SAVE_IMAGE)
        ctx = builder.build_context(nd, inputs)
        self.assertIsInstance(ctx, NodeContext)


# ── QwenEditContextBuilder ───────────────────────────────────────────────


class TestQwenEditContextBuilder(unittest.TestCase):
    """QwenEditContextBuilder 的 prepare / build / condition。"""

    def _prepare(self, extra_inputs=None):
        """辅助方法：创建 builder 并调用 prepare_generate_inputs。"""
        from kdit.pipelines.context_builders.qwen import QwenEditContextBuilder

        builder = QwenEditContextBuilder()
        inputs = _make_inputs()
        inputs.runtime_config.size = (1024, 1024)
        settings = _make_qwen_settings()
        builder.prepare_generate_inputs(
            inputs,
            extra_inputs,
            _default_settings=settings,
            _engine=MagicMock(),
            _vae_model_key=ModelKey.QwenImageVAE,
        )
        return builder, inputs

    def test_prepare_with_no_image(self):
        """无参考图时 img_path 为 None，target_h/w 有值。"""
        builder, _ = self._prepare()

        self.assertIsNone(builder._extra.img_path)
        self.assertEqual(builder._extra.target_h, 1024)
        self.assertEqual(builder._extra.target_w, 1024)

    def test_has_ref_images_false_when_no_image(self):
        """无参考图时 has_ref_images 返回 False。"""
        builder, inputs = self._prepare()

        self.assertFalse(builder.has_ref_images(inputs))

    def test_prepare_with_ref_images(self):
        """有参考图时 img_path 不为 None。"""
        extra = QwenEditExtraInputs(img_path="ref.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        self.assertIsNotNone(builder._extra.img_path)
        self.assertTrue(builder.has_ref_images(inputs))

    def test_build_context_text_encode_with_condition_image(self):
        """Edit 模式 build_context(TEXT_ENCODE) 包含 condition_image_path。"""
        extra = QwenEditExtraInputs(img_path="ref.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        nd = _node_def(NT.TEXT_ENCODE, ModelKey.Qwen2VLTextEncoderMultimodal)
        ctx = builder.build_context(nd, inputs)

        self.assertIn("condition_image_path", ctx.metadata)

    def test_build_context_read_image(self):
        """build_context(READ_IMAGE) 返回包含 img_paths 的 context。"""
        extra = QwenEditExtraInputs(img_path="ref.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        nd = _node_def(NT.READ_IMAGE, node_id=10)
        ctx = builder.build_context(nd, inputs)

        self.assertIn("img_paths", ctx.metadata)

    def test_build_context_vae_encode_images(self):
        """build_context(VAE_ENCODE_IMAGES) 返回空 context。"""
        extra = QwenEditExtraInputs(img_path="ref.png")
        builder, inputs = self._prepare(extra_inputs=extra)

        nd = _node_def(NT.VAE_ENCODE_IMAGES, ModelKey.QwenImageVAE)
        ctx = builder.build_context(nd, inputs)

        self.assertIsInstance(ctx, NodeContext)

    def test_build_context_unexpected_type_raises(self):
        """build_context 遇到未知 node_type 时抛出 ValueError。"""
        builder, inputs = self._prepare()

        nd = _node_def(NT.VAE_ENCODE_SPATIAL)
        with self.assertRaises(ValueError, msg="unexpected node_type"):
            builder.build_context(nd, inputs)


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
