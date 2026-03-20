# Copyright 2025 Tencent
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

"""Tests for kdit.generators.generator_context — GeneratorInferContext dataclass。"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.config import RuntimeConfig, SampleConfig, SolverType
from kdit.generators.generator_context import GeneratorInferContext


class TestGeneratorRunContext(unittest.TestCase):
    """GeneratorInferContext 是一个 dataclass，验证字段默认值和赋值。"""

    def test_default_values(self):
        ctx = GeneratorInferContext()
        self.assertIsNone(ctx.diffusion_model)
        self.assertIsNone(ctx.positive)
        self.assertIsNone(ctx.negative)
        self.assertIsNone(ctx.base_latent)
        self.assertIsNone(ctx.aux_latent)
        self.assertIsNone(ctx.device)
        self.assertIsNone(ctx.offload_device)
        self.assertIsNone(ctx.sample_config)
        self.assertIsNone(ctx.runtime_config)
        self.assertIsNone(ctx.cache_config)
        self.assertIsNone(ctx.video_control)
        self.assertIsNone(ctx.control_video_config)
        self.assertIsNone(ctx.comfy_bar_callback)

    def test_full_construction(self):
        mock_model = MagicMock()
        positive = torch.randn(1, 77, 768)
        negative = torch.randn(1, 77, 768)
        sample_cfg = SampleConfig(steps=20, cfg_scale=5.0, solver=SolverType.EULER)
        runtime_cfg = RuntimeConfig(size=(512, 512), frame_num=16, seed=42)

        ctx = GeneratorInferContext(
            diffusion_model=mock_model,
            positive=positive,
            negative=negative,
            base_latent=None,
            aux_latent=None,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            sample_config=sample_cfg,
            runtime_config=runtime_cfg,
            cache_config=None,
            video_control=None,
            control_video_config=None,
            comfy_bar_callback=None,
        )

        self.assertIs(ctx.diffusion_model, mock_model)
        self.assertTrue(torch.equal(ctx.positive, positive))
        self.assertEqual(ctx.device, torch.device("cpu"))
        self.assertIs(ctx.sample_config, sample_cfg)
        self.assertIs(ctx.runtime_config, runtime_cfg)

    def test_is_dataclass(self):
        import dataclasses

        self.assertTrue(dataclasses.is_dataclass(GeneratorInferContext))

    def test_comfy_bar_callback_accepts_callable(self):
        callback = MagicMock()
        ctx = GeneratorInferContext(comfy_bar_callback=callback)
        self.assertIs(ctx.comfy_bar_callback, callback)


if __name__ == "__main__":
    unittest.main()
