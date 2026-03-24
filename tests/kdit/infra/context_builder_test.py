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

"""ContextBuilder 基类单元测试。"""

import unittest
from unittest.mock import MagicMock

from kdit.nodes.core.node_context import NodeContext
from kdit.pipelines.context_builder import ContextBuilder
from kdit.pipelines.generate_inputs import PipelineGenerateInputs


class _DummyContextBuilder(ContextBuilder):
    """最小可用的 ContextBuilder — 用于基类测试。"""

    def build_context(self, phase, inputs):
        return NodeContext()


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

    def test_post_process_default_passthrough(self):
        """默认 post_process 直接返回输入。"""
        builder = _DummyContextBuilder()
        sentinel = object()
        result = builder.post_process(sentinel, self._make_inputs())
        self.assertIs(result, sentinel)

    def test_common_metadata(self):
        """_common_metadata 包含 offload_model。"""
        inputs = self._make_inputs()
        inputs.runtime_config.offload_model = True
        meta = ContextBuilder._common_metadata(inputs)
        self.assertTrue(meta["offload_model"])


if __name__ == "__main__":
    unittest.main()
