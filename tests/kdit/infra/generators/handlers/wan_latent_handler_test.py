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

"""WanLatentHandler 单元测试。

测试覆盖:
- preprocess_base (T2V / I2V concat / 单元素 / 空 list)
- validate_noise_shape (T2V / I2V z_dim 覆写)
- apply_aux_latent (None / 非 5D / 噪声混合 / 无噪声 / 帧填充)
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.generators.handlers_impl.wan_latent import WanLatentHandler
from kdit.models.model_key import ModelKey


class TestWanPreprocessBase(unittest.TestCase):
    """测试 preprocess_base — I2V concat 与 T2V 直通。"""

    def test_t2v_returns_first_element(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        latent = torch.randn(1, 16, 21, 30, 52)
        result = handler.preprocess_base([latent])
        self.assertTrue(torch.equal(result, latent))

    def test_i2v_concat_shape(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_I2V_14B)
        latent = torch.randn(1, 16, 21, 30, 52)
        mask = torch.randn(1, 1, 21, 30, 52)
        result = handler.preprocess_base([latent, mask])
        self.assertEqual(result.shape, (1, 17, 21, 30, 52))

    def test_i2v_concat_order_mask_before_latent(self):
        """回归测试：preprocess_base 必须按 [mask, latent] 顺序 concat。

        Wan I2V 模型要求 channel 维度前 1 通道为 mask、后 16 通道为 latent。
        此测试确保 concat 顺序不会被意外改回 [latent, mask]。
        """
        handler = WanLatentHandler(ModelKey.Wan2_2_I2V_14B)
        latent = torch.ones(1, 16, 2, 4, 4)  # 全 1
        mask = torch.zeros(1, 1, 2, 4, 4)  # 全 0
        result = handler.preprocess_base([latent, mask])
        # mask（全 0）应在 channel 维度前面，latent（全 1）在后面
        mask_part = result[:, :1, :, :, :]
        latent_part = result[:, 1:, :, :, :]
        self.assertTrue(torch.equal(mask_part, mask), "前 1 通道应为 mask（全 0）")
        self.assertTrue(torch.equal(latent_part, latent), "后 16 通道应为 latent（全 1）")

    def test_i2v_single_element_returns_directly(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_I2V_14B)
        latent = torch.randn(1, 16, 21, 30, 52)
        result = handler.preprocess_base([latent])
        self.assertTrue(torch.equal(result, latent))

    def test_empty_list_raises(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_I2V_14B)
        with self.assertRaises((ValueError, IndexError)):
            handler.preprocess_base([])


class TestWanValidateNoiseShape(unittest.TestCase):
    """测试 validate_noise_shape — 校验 noise_shape。"""

    def test_t2v_keeps_original_shape(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        dm = MagicMock()
        result = handler.validate_noise_shape([16, 8, 32, 32], dm, ModelKey.Wan2_2_T2V_14B)
        self.assertEqual(result, [16, 8, 32, 32])

    def test_i2v_overrides_z_dim(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_I2V_14B)
        dm = MagicMock()
        result = handler.validate_noise_shape([16, 8, 32, 32], dm, ModelKey.Wan2_2_I2V_14B)
        # validate_noise_shape 只做校验，不覆写 z_dim
        self.assertEqual(result, [16, 8, 32, 32])


class TestWanApplyAuxLatent(unittest.TestCase):
    """测试 apply_aux_latent — 噪声混合逻辑。"""

    def _make_sample_config(self, add_noise=True):
        sc = MagicMock()
        sc.add_noise_to_latent = add_noise
        return sc

    def test_none_input_returns_noise(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        noise = torch.randn(1, 16, 4, 8, 8)
        result = handler.apply_aux_latent(
            noise,
            None,
            self._make_sample_config(),
            torch.tensor([500]),
            1000,
        )
        self.assertTrue(torch.equal(result, noise))

    def test_non_5d_raises(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        noise = torch.randn(1, 16, 8, 8)  # 4D
        with self.assertRaises(ValueError):
            handler.apply_aux_latent(
                noise,
                torch.randn(1, 16, 8, 8),
                self._make_sample_config(),
                torch.tensor([500]),
                1000,
            )

    def test_add_noise_blending(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        noise = torch.ones(1, 16, 4, 8, 8)
        input_lat = torch.zeros(1, 16, 4, 8, 8)
        timesteps = torch.tensor([500])
        result = handler.apply_aux_latent(
            noise,
            input_lat,
            self._make_sample_config(True),
            timesteps,
            1000,
        )
        # result = noise * (t/T) + (1 - t/T) * input = 0.5
        expected = torch.full_like(result, 0.5)
        self.assertTrue(torch.allclose(result, expected))

    def test_no_noise_uses_input_directly(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        noise = torch.ones(1, 16, 4, 8, 8)
        input_lat = torch.full((1, 16, 4, 8, 8), 2.0)
        result = handler.apply_aux_latent(
            noise,
            input_lat,
            self._make_sample_config(False),
            torch.tensor([500]),
            1000,
        )
        self.assertTrue(torch.equal(result, input_lat))

    def test_frame_padding(self):
        handler = WanLatentHandler(ModelKey.Wan2_2_T2V_14B)
        noise = torch.randn(1, 16, 8, 8, 8)  # 8 frames
        input_lat = torch.randn(1, 16, 4, 8, 8)  # 4 frames
        result = handler.apply_aux_latent(
            noise,
            input_lat,
            self._make_sample_config(False),
            torch.tensor([500]),
            1000,
        )
        self.assertEqual(result.shape, noise.shape)


if __name__ == "__main__":
    unittest.main()
