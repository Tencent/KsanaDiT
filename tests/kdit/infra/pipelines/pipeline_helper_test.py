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

"""Pipeline 辅助函数和 __init__.py 导出测试。"""

import unittest
from unittest.mock import MagicMock

from kdit.config.sample_config import SampleConfig
from kdit.pipelines.generate_inputs import PipelineGenerateInputs
from kdit.pipelines.pipeline import (
    _ensure_cache_config_list,
    _get_num_prompts,
    _merge_sample_config,
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
