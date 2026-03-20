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

"""QwenDenoiseHandler 单元测试。

测试覆盖:
- prepare_model_forward_kargs (combine / split / no-cfg / cache raises)
- apply_cfg (标准 CFG / Edit 模式 norm rescale)
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.generators.handlers_impl.qwen_denoise import QwenDenoiseHandler
from kdit.generators.handlers_impl.qwen_latent import QwenLatentHandler
from kdit.models.model_key import ModelKey


class TestQwenPrepareModelForwardKargs(unittest.TestCase):
    """测试 prepare_model_forward_kargs — combine/split/no-cfg。"""

    def _make_handler_and_args(self, cfg_scale=7.5, combine=True):
        latent_handler = QwenLatentHandler()
        handler = QwenDenoiseHandler(ModelKey.QwenImage_T2I, latent_handler)
        # 先 pack 以初始化 _latent_img_shapes
        latent_handler.pack_noise(torch.randn(2, 16, 1, 32, 32), 2)
        noise = torch.randn(2, 256, 64)
        timestep = torch.tensor([500])
        pos_embeds = torch.randn(2, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_embeds = torch.randn(2, 10, 768)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        kwargs = {
            "cfg_scale": cfg_scale,
            "noise_latent": noise,
            "timestep": timestep,
            "combine_cond_uncond": combine,
            "step_iter": 0,
            "cache": None,
            "positive": (pos_embeds, pos_mask),
            "negative": (neg_embeds, neg_mask),
            "base_latent": None,
        }
        return handler, kwargs

    def test_combine_mode(self):
        handler, kwargs = self._make_handler_and_args(cfg_scale=7.5, combine=True)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "combine")
        self.assertEqual(result["x"].shape[0], 4)

    def test_split_mode(self):
        handler, kwargs = self._make_handler_and_args(cfg_scale=7.5, combine=False)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["phase"], "cond")
        self.assertEqual(result[1]["phase"], "uncond")

    def test_no_cfg(self):
        handler, kwargs = self._make_handler_and_args(cfg_scale=1.0, combine=False)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "cond")

    def test_cache_raises(self):
        handler, kwargs = self._make_handler_and_args()
        kwargs["cache"] = MagicMock()
        with self.assertRaises(NotImplementedError):
            handler.prepare_model_forward_kargs(**kwargs)


class TestQwenApplyCfg(unittest.TestCase):
    """测试 apply_cfg — 标准 CFG 和 Edit 模式。"""

    def test_standard_cfg(self):
        latent_handler = QwenLatentHandler()
        handler = QwenDenoiseHandler(ModelKey.QwenImage_T2I, latent_handler)
        cond = torch.ones(2, 4)
        uncond = torch.zeros(2, 4)
        result = handler.apply_cfg(7.5, cond, uncond)
        expected = uncond + 7.5 * (cond - uncond)
        self.assertTrue(torch.allclose(result, expected))

    def test_edit_cfg_with_norm_rescale(self):
        latent_handler = QwenLatentHandler()
        handler = QwenDenoiseHandler(ModelKey.QwenImage_Edit, latent_handler)
        cond = torch.randn(2, 4)
        uncond = torch.randn(2, 4)
        result = handler.apply_cfg(7.5, cond, uncond)
        # Edit 模式会做 norm rescale，结果形状应一致
        self.assertEqual(result.shape, cond.shape)


if __name__ == "__main__":
    unittest.main()
