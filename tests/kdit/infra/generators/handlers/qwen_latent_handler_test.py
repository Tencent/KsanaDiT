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

"""QwenLatentHandler 单元测试。

测试覆盖:
- pack_noise / unpack_noise (5D ↔ 3D 变换)
- _pack_aux_latents (参考图 latent 打包 + img_shapes 更新)
- maybe_update_sample_config (shift 自动计算)
- calculate_shift (线性插值)
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.generators.handlers_impl.qwen_latent import QwenLatentHandler


class TestQwenPackUnpackNoise(unittest.TestCase):
    """测试 pack_noise / unpack_noise 往返一致性。"""

    def test_pack_shape(self):
        handler = QwenLatentHandler()
        # [num, z_dim, 1, h, w] — latent_f 必须为 1
        latents = torch.randn(2, 16, 1, 32, 32)
        patch_size = 2
        packed = handler.pack_noise(latents, patch_size)
        # packed: [2, (32/2)*(32/2), 16*2*2] = [2, 256, 64]
        self.assertEqual(packed.shape, (2, 256, 64))

    def test_pack_non_5d_raises(self):
        handler = QwenLatentHandler()
        with self.assertRaises(ValueError):
            handler.pack_noise(torch.randn(2, 16, 32, 32), 2)

    def test_roundtrip(self):
        handler = QwenLatentHandler()
        latents = torch.randn(2, 16, 1, 32, 32)
        patch_size = 2
        packed = handler.pack_noise(latents, patch_size)
        unpacked = handler.unpack_noise(packed, patch_size)
        self.assertEqual(unpacked.shape, latents.shape)
        self.assertTrue(torch.allclose(unpacked, latents, atol=1e-6))

    def test_unpack_non_3d_raises(self):
        handler = QwenLatentHandler()
        # 先 pack 以初始化 _latent_img_shapes
        handler.pack_noise(torch.randn(2, 16, 1, 32, 32), 2)
        with self.assertRaises(ValueError):
            handler.unpack_noise(torch.randn(2, 256), 2)


class TestQwenPackAuxLatents(unittest.TestCase):
    """测试 _pack_aux_latents — 参考图 latent 打包。"""

    def test_pack_updates_img_shapes(self):
        handler = QwenLatentHandler()
        # 先 pack noise 以初始化 _latent_img_shapes
        handler.pack_noise(torch.randn(2, 16, 1, 32, 32), 2)
        ref = torch.randn(2, 16, 1, 16, 16)
        packed = handler._pack_aux_latents([ref], patch_size=2)
        self.assertEqual(len(packed), 1)
        # 每个 batch 的 img_shapes 应多一个 ref shape
        self.assertEqual(len(handler._latent_img_shapes[0]), 2)


class TestQwenMaybeUpdateSampleConfig(unittest.TestCase):
    """测试 maybe_update_sample_config — shift 自动计算。"""

    def test_existing_shift_unchanged(self):
        handler = QwenLatentHandler()
        sc = MagicMock()
        sc.shift = 0.8
        result = handler.maybe_update_sample_config(sc, [2, 256, 64], MagicMock())
        self.assertIs(result, sc)

    def test_none_shift_computes(self):
        from kdit.config.sample_config import SampleConfig

        handler = QwenLatentHandler()
        sc = SampleConfig(shift=None)
        default_settings = MagicMock()
        ds_sc = default_settings.sample_config
        ds_sc.base_seq_len = 256
        ds_sc.max_seq_len = 4096
        ds_sc.base_shift = 0.5
        ds_sc.max_shift = 1.15
        result = handler.maybe_update_sample_config(sc, [2, 256, 64], default_settings)
        self.assertIsNotNone(result.shift)


class TestQwenCalculateShift(unittest.TestCase):
    """测试 calculate_shift — 线性插值。"""

    def test_base_seq_len(self):
        handler = QwenLatentHandler()
        configs = MagicMock()
        configs.base_seq_len = 256
        configs.max_seq_len = 4096
        configs.base_shift = 0.5
        configs.max_shift = 1.15
        result = handler.calculate_shift(256, configs)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_max_seq_len(self):
        handler = QwenLatentHandler()
        configs = MagicMock()
        configs.base_seq_len = 256
        configs.max_seq_len = 4096
        configs.base_shift = 0.5
        configs.max_shift = 1.15
        result = handler.calculate_shift(4096, configs)
        self.assertAlmostEqual(result, 1.15, places=4)


if __name__ == "__main__":
    unittest.main()
