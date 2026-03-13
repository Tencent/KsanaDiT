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

"""Tests for kdit.generators.steps.validation — 5 个验证函数。"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from kdit.config.cache_config import KsanaHybridCacheConfig, KsanaStepCacheConfig
from kdit.generators.steps.validation import (
    valid_cache_config,
    valid_diffusion_model,
    valid_input_latent,
    valid_runtime_config,
    valid_sample_config,
)
from kdit.models import KsanaDiffusionModel
from kdit.models.model_key import KsanaModelKey


def _make_mock_diffusion_model(model_key=KsanaModelKey.Wan2_2_T2V_14B, run_dtype=torch.float16):
    """创建一个 mock KsanaDiffusionModel。"""
    m = MagicMock(spec=KsanaDiffusionModel)
    m.model_key = model_key
    m.run_dtype = run_dtype
    m.default_settings = {"num_train_timesteps": 1000}
    return m


class TestValidDiffusionModel(unittest.TestCase):
    """valid_diffusion_model 校验并规范化为 list。"""

    def test_single_model_wrapped_to_list(self):
        model = _make_mock_diffusion_model()
        result = valid_diffusion_model(model, KsanaModelKey.Wan2_2_T2V_14B)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_list_of_one_model_passes(self):
        model = _make_mock_diffusion_model()
        result = valid_diffusion_model([model], KsanaModelKey.Wan2_2_T2V_14B)
        self.assertEqual(len(result), 1)

    def test_two_models_for_wan_passes(self):
        m1 = _make_mock_diffusion_model(KsanaModelKey.Wan2_2_I2V_14B)
        m2 = _make_mock_diffusion_model(KsanaModelKey.Wan2_2_I2V_14B)
        result = valid_diffusion_model([m1, m2], KsanaModelKey.Wan2_2_I2V_14B)
        self.assertEqual(len(result), 2)

    def test_two_models_for_non_wan_raises(self):
        m1 = _make_mock_diffusion_model(KsanaModelKey.QwenImage_T2I)
        m2 = _make_mock_diffusion_model(KsanaModelKey.QwenImage_T2I)
        with self.assertRaises(ValueError):
            valid_diffusion_model([m1, m2], KsanaModelKey.QwenImage_T2I)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            valid_diffusion_model("not_a_model", KsanaModelKey.Wan2_2_T2V_14B)

    def test_two_models_mismatched_dtype_raises(self):
        m1 = _make_mock_diffusion_model(KsanaModelKey.Wan2_2_I2V_14B, run_dtype=torch.float16)
        m2 = _make_mock_diffusion_model(KsanaModelKey.Wan2_2_I2V_14B, run_dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            valid_diffusion_model([m1, m2], KsanaModelKey.Wan2_2_I2V_14B)


class TestValidSampleConfig(unittest.TestCase):
    """valid_sample_config 校验 cfg_scale / solver / denoise。"""

    def _make_config(self, cfg_scale=5.0, solver=KsanaSolverType.EULER, denoise=1.0, steps=20):
        return KsanaSampleConfig(steps=steps, cfg_scale=cfg_scale, solver=solver, denoise=denoise)

    def test_scalar_cfg_scale_expanded_to_list(self):
        cfg = self._make_config(cfg_scale=5.0)
        result = valid_sample_config(cfg, model_len=2)
        self.assertIsInstance(result.cfg_scale, list)
        self.assertEqual(len(result.cfg_scale), 2)
        self.assertEqual(result.cfg_scale, [5.0, 5.0])

    def test_list_cfg_scale_passes(self):
        cfg = self._make_config(cfg_scale=[3.0, 5.0])
        result = valid_sample_config(cfg, model_len=2)
        self.assertEqual(result.cfg_scale, [3.0, 5.0])

    def test_short_list_cfg_scale_raises(self):
        cfg = self._make_config(cfg_scale=[3.0])
        with self.assertRaises(ValueError):
            valid_sample_config(cfg, model_len=2)

    def test_invalid_solver_raises(self):
        cfg = self._make_config(solver=None)
        with self.assertRaises(ValueError):
            valid_sample_config(cfg, model_len=1)

    def test_zero_denoise_raises(self):
        cfg = self._make_config(denoise=0.0)
        with self.assertRaises(ValueError):
            valid_sample_config(cfg, model_len=1)

    def test_negative_denoise_raises(self):
        cfg = self._make_config(denoise=-0.5)
        with self.assertRaises(ValueError):
            valid_sample_config(cfg, model_len=1)


class TestValidCacheConfig(unittest.TestCase):
    """valid_cache_config 校验并转换为 HybridCacheConfig list。"""

    def test_none_returns_none(self):
        self.assertIsNone(valid_cache_config(None, model_len=1))

    def test_single_step_cache_wrapped(self):
        step_cache = KsanaStepCacheConfig(name="teacache")
        result = valid_cache_config([step_cache], model_len=1)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], KsanaHybridCacheConfig)

    def test_length_mismatch_raises(self):
        step_cache = KsanaStepCacheConfig(name="teacache")
        with self.assertRaises(ValueError):
            valid_cache_config([step_cache, step_cache, step_cache], model_len=2)

    def test_none_element_preserved(self):
        result = valid_cache_config([None], model_len=1)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0])

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            valid_cache_config(["not_a_config"], model_len=1)


class TestValidRuntimeConfig(unittest.TestCase):
    """valid_runtime_config 校验 batch_size_per_prompts。"""

    def _make_config(self, batch_size_per_prompts=None, seed=42):
        return KsanaRuntimeConfig(
            size=(512, 512),
            frame_num=16,
            batch_size_per_prompts=batch_size_per_prompts,
            seed=seed,
        )

    def test_none_batch_size_defaults_to_ones(self):
        cfg = self._make_config(batch_size_per_prompts=None)
        result = valid_runtime_config(cfg, num_prompts=3)
        self.assertEqual(result.batch_size_per_prompts, [1, 1, 1])

    def test_int_batch_size_expanded(self):
        cfg = self._make_config(batch_size_per_prompts=2)
        result = valid_runtime_config(cfg, num_prompts=3)
        self.assertEqual(result.batch_size_per_prompts, [2, 2, 2])

    def test_list_batch_size_passes(self):
        cfg = self._make_config(batch_size_per_prompts=[1, 2, 3])
        result = valid_runtime_config(cfg, num_prompts=3)
        self.assertEqual(result.batch_size_per_prompts, [1, 2, 3])

    def test_list_batch_size_wrong_length_raises(self):
        cfg = self._make_config(batch_size_per_prompts=[1, 2])
        with self.assertRaises(ValueError):
            valid_runtime_config(cfg, num_prompts=3)

    def test_none_config_raises(self):
        with self.assertRaises(ValueError):
            valid_runtime_config(None, num_prompts=1)


class TestValidInputLatent(unittest.TestCase):
    """valid_input_latent 校验 input_latent 与 noise_shape 的维度一致性。"""

    def test_none_input_latent_passes(self):
        # 不应抛异常
        valid_input_latent(None, (1, 4, 16, 32, 32))

    def test_matching_shape_passes(self):
        latent = torch.zeros(1, 4, 16, 32, 32)
        valid_input_latent(latent, (1, 4, 16, 32, 32))

    def test_different_frame_dim_passes(self):
        """frame 维度可以不同。"""
        latent = torch.zeros(1, 4, 8, 32, 32)
        valid_input_latent(latent, (1, 4, 16, 32, 32))

    def test_mismatched_batch_raises(self):
        latent = torch.zeros(2, 4, 16, 32, 32)
        with self.assertRaises(ValueError):
            valid_input_latent(latent, (1, 4, 16, 32, 32))

    def test_mismatched_spatial_raises(self):
        latent = torch.zeros(1, 4, 16, 64, 64)
        with self.assertRaises(ValueError):
            valid_input_latent(latent, (1, 4, 16, 32, 32))

    def test_4d_tensor_raises(self):
        latent = torch.zeros(1, 4, 32, 32)
        with self.assertRaises(ValueError):
            valid_input_latent(latent, (1, 4, 16, 32, 32))


if __name__ == "__main__":
    unittest.main()
