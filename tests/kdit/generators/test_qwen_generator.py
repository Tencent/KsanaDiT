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

"""QwenGenerator 钩子方法单元测试。

测试覆盖:
- preprocess_text_conditioning (tuple / bare tensor / invalid)
- _valid_prompts (mask 维度校验)
- cast_text_tensors_to (embeds + mask 设备/dtype 转换)
- apply_cfg (标准 CFG / Edit 模式 norm rescale)
- calculate_shift (线性插值)
- pack_noise_latents / unpack_noise_latents (5D ↔ 3D 变换)
- pack_ref_latents (参考图 latent 打包 + img_shapes 更新)
- maybe_update_sample_config (shift 自动计算)
- _pad_text_pair (不等长 padding)
- prepare_model_forward_kargs (combine / split / no-cfg)
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.models import KsanaModelKey


def _make_qwen_generator(model_key=KsanaModelKey.QwenImage_T2I):
    """构造 QwenGenerator 实例，绕过 AdvancedFactory 注册。"""
    from kdit.generators.qwen_generator import QwenGenerator

    gen = QwenGenerator.__new__(QwenGenerator)
    gen.__init__()
    gen.model_key = model_key
    return gen


class TestPreprocessTextConditioning(unittest.TestCase):
    """测试 preprocess_text_conditioning。"""

    def test_tuple_passthrough(self):
        gen = _make_qwen_generator()
        embeds = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        result = gen.preprocess_text_conditioning((embeds, mask))
        self.assertTrue(torch.equal(result[0], embeds))
        self.assertTrue(torch.equal(result[1], mask))

    def test_bare_tensor_generates_mask(self):
        gen = _make_qwen_generator()
        embeds = torch.randn(2, 10, 768)
        result = gen.preprocess_text_conditioning(embeds)
        self.assertEqual(result[0].shape, embeds.shape)
        self.assertEqual(result[1].shape, (2, 10))
        self.assertTrue((result[1] == 1).all())

    def test_invalid_format_raises(self):
        gen = _make_qwen_generator()
        with self.assertRaises(ValueError, msg="Unsupported conditioning format"):
            gen.preprocess_text_conditioning(42)


class TestQwenValidPrompts(unittest.TestCase):
    """测试 _valid_prompts — mask 维度校验。"""

    def test_valid_prompts(self):
        gen = _make_qwen_generator()
        pos = torch.randn(2, 10, 768)
        neg = torch.randn(2, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        result_pos, _ = gen._valid_prompts((pos, pos_mask), (neg, neg_mask))
        self.assertEqual(result_pos[0].shape[0], 2)

    def test_mismatched_batch_raises(self):
        gen = _make_qwen_generator()
        pos = torch.randn(2, 10, 768)
        neg = torch.randn(3, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(3, 10, dtype=torch.long)
        with self.assertRaises(ValueError):
            gen._valid_prompts((pos, pos_mask), (neg, neg_mask))


class TestQwenCastTextTensors(unittest.TestCase):
    """测试 cast_text_tensors_to — embeds + mask 转换。"""

    def test_cast_dtype_and_device(self):
        gen = _make_qwen_generator()
        pos = torch.randn(2, 10, 768, dtype=torch.float32)
        neg = torch.randn(2, 10, 768, dtype=torch.float32)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        result_pos, result_neg = gen.cast_text_tensors_to(
            (pos, pos_mask), (neg, neg_mask), dtype=torch.float16, device=torch.device("cpu")
        )
        self.assertEqual(result_pos[0].dtype, torch.float16)
        self.assertEqual(result_neg[0].dtype, torch.float16)
        # mask 应在同一设备上
        self.assertEqual(result_pos[1].device, torch.device("cpu"))


class TestQwenApplyCfg(unittest.TestCase):
    """测试 apply_cfg — 标准 CFG 和 Edit 模式。"""

    def test_standard_cfg(self):
        gen = _make_qwen_generator(KsanaModelKey.QwenImage_T2I)
        cond = torch.ones(2, 4)
        uncond = torch.zeros(2, 4)
        result = gen.apply_cfg(7.5, cond, uncond)
        expected = uncond + 7.5 * (cond - uncond)
        self.assertTrue(torch.allclose(result, expected))

    def test_edit_cfg_with_norm_rescale(self):
        gen = _make_qwen_generator(KsanaModelKey.QwenImage_Edit)
        cond = torch.randn(2, 4)
        uncond = torch.randn(2, 4)
        result = gen.apply_cfg(7.5, cond, uncond)
        # Edit 模式会做 norm rescale，结果形状应一致
        self.assertEqual(result.shape, cond.shape)


class TestCalculateShift(unittest.TestCase):
    """测试 calculate_shift — 线性插值。"""

    def test_base_seq_len(self):
        gen = _make_qwen_generator()
        configs = MagicMock()
        configs.base_seq_len = 256
        configs.max_seq_len = 4096
        configs.base_shift = 0.5
        configs.max_shift = 1.15
        # 在 base_seq_len 处应返回 base_shift
        result = gen.calculate_shift(256, configs)
        self.assertAlmostEqual(result, 0.5, places=4)

    def test_max_seq_len(self):
        gen = _make_qwen_generator()
        configs = MagicMock()
        configs.base_seq_len = 256
        configs.max_seq_len = 4096
        configs.base_shift = 0.5
        configs.max_shift = 1.15
        result = gen.calculate_shift(4096, configs)
        self.assertAlmostEqual(result, 1.15, places=4)


class TestPackUnpackNoiseLatents(unittest.TestCase):
    """测试 pack_noise_latents / unpack_noise_latents 往返一致性。"""

    def test_pack_shape(self):
        gen = _make_qwen_generator()
        # [num, z_dim, 1, h, w] — latent_f 必须为 1
        latents = torch.randn(2, 16, 1, 32, 32)
        patch_size = 2
        packed = gen.pack_noise_latents(latents, patch_size)
        # packed: [2, (32/2)*(32/2), 16*2*2] = [2, 256, 64]
        self.assertEqual(packed.shape, (2, 256, 64))

    def test_pack_non_5d_raises(self):
        gen = _make_qwen_generator()
        with self.assertRaises(ValueError, msg="must be 5D"):
            gen.pack_noise_latents(torch.randn(2, 16, 32, 32), 2)

    def test_pack_latent_f_not_1_raises(self):
        gen = _make_qwen_generator()
        with self.assertRaises(ValueError, msg="latent_f  must be 1"):
            gen.pack_noise_latents(torch.randn(2, 16, 4, 32, 32), 2)

    def test_roundtrip(self):
        gen = _make_qwen_generator()
        latents = torch.randn(2, 16, 1, 32, 32)
        patch_size = 2
        packed = gen.pack_noise_latents(latents, patch_size)
        unpacked = gen.unpack_noise_latents(packed, patch_size)
        self.assertEqual(unpacked.shape, latents.shape)
        self.assertTrue(torch.allclose(unpacked, latents, atol=1e-6))

    def test_unpack_non_3d_raises(self):
        gen = _make_qwen_generator()
        gen.latent_img_shapes = [[(1, 16, 16)] for _ in range(2)]
        with self.assertRaises(ValueError, msg="must be 3D"):
            gen.unpack_noise_latents(torch.randn(2, 256), 2)


class TestPackRefLatents(unittest.TestCase):
    """测试 pack_ref_latents — 参考图 latent 打包。"""

    def test_pack_updates_img_shapes(self):
        gen = _make_qwen_generator()
        # 先 pack noise 以初始化 latent_img_shapes
        gen.pack_noise_latents(torch.randn(2, 16, 1, 32, 32), 2)
        ref = torch.randn(2, 16, 1, 16, 16)
        packed = gen.pack_ref_latents([ref], patch_size=2)
        self.assertEqual(len(packed), 1)
        # 每个 batch 的 img_shapes 应多一个 ref shape
        self.assertEqual(len(gen.latent_img_shapes[0]), 2)


class TestMaybeUpdateSampleConfig(unittest.TestCase):
    """测试 maybe_update_sample_config — shift 自动计算。"""

    def test_existing_shift_unchanged(self):
        gen = _make_qwen_generator()
        sc = MagicMock()
        sc.shift = 0.8
        result = gen.maybe_update_sample_config(sc, [2, 256, 64], MagicMock())
        self.assertIs(result, sc)

    def test_none_shift_computes(self):
        from kdit.config import KsanaSampleConfig

        gen = _make_qwen_generator()
        sc = KsanaSampleConfig(shift=None)
        default_settings = MagicMock()
        default_settings.sample_config.base_seq_len = 256
        default_settings.sample_config.max_seq_len = 4096
        default_settings.sample_config.base_shift = 0.5
        default_settings.sample_config.max_shift = 1.15
        result = gen.maybe_update_sample_config(sc, [2, 256, 64], default_settings)
        self.assertIsNotNone(result.shift)

    def test_non_3d_shape_raises(self):
        from kdit.config import KsanaSampleConfig

        gen = _make_qwen_generator()
        sc = KsanaSampleConfig(shift=None)
        with self.assertRaises(RuntimeError, msg="should be 3D"):
            gen.maybe_update_sample_config(sc, [2, 256], MagicMock())


class TestPadTextPair(unittest.TestCase):
    """测试 _pad_text_pair — 不等长 padding。"""

    def test_equal_length_no_pad(self):
        gen = _make_qwen_generator()
        a = torch.randn(2, 10, 768)
        ma = torch.ones(2, 10)
        b = torch.randn(2, 10, 768)
        mb = torch.ones(2, 10)
        ra, _, rb, _ = gen._pad_text_pair(a, ma, b, mb)
        self.assertEqual(ra.shape[1], 10)
        self.assertEqual(rb.shape[1], 10)

    def test_unequal_length_pads(self):
        gen = _make_qwen_generator()
        a = torch.randn(2, 8, 768)
        ma = torch.ones(2, 8)
        b = torch.randn(2, 12, 768)
        mb = torch.ones(2, 12)
        ra, rma, rb, _ = gen._pad_text_pair(a, ma, b, mb)
        self.assertEqual(ra.shape[1], 12)
        self.assertEqual(rb.shape[1], 12)
        self.assertEqual(rma.shape[1], 12)


class TestQwenPrepareModelForwardKargs(unittest.TestCase):
    """测试 prepare_model_forward_kargs — combine/split/no-cfg。"""

    def _make_args(self, cfg_scale=7.5, combine=True):
        gen = _make_qwen_generator()
        # 先 pack 以初始化 latent_img_shapes
        gen.pack_noise_latents(torch.randn(2, 16, 1, 32, 32), 2)
        noise = torch.randn(2, 256, 64)
        timestep = torch.tensor([500])
        pos_embeds = torch.randn(2, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_embeds = torch.randn(2, 10, 768)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        return gen, {
            "cfg_scale": cfg_scale,
            "noise_latent": noise,
            "timestep": timestep,
            "combine_cond_uncond": combine,
            "step_iter": 0,
            "cache": None,
            "positive": (pos_embeds, pos_mask),
            "negative": (neg_embeds, neg_mask),
            "image_embeds": None,
        }

    def test_combine_mode(self):
        gen, kwargs = self._make_args(cfg_scale=7.5, combine=True)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "combine")
        self.assertEqual(result["x"].shape[0], 4)

    def test_split_mode(self):
        gen, kwargs = self._make_args(cfg_scale=7.5, combine=False)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_no_cfg(self):
        gen, kwargs = self._make_args(cfg_scale=1.0, combine=False)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "cond")

    def test_cache_raises(self):
        gen, kwargs = self._make_args()
        kwargs["cache"] = MagicMock()
        with self.assertRaises(NotImplementedError):
            gen.prepare_model_forward_kargs(**kwargs)


if __name__ == "__main__":
    unittest.main()
