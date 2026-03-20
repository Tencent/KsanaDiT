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

"""Tests for kdit.generators.steps.noise — create_random_noise_latents / create_cache."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.config import RuntimeConfig
from kdit.config.cache_config import StepCacheConfig
from kdit.generators.steps.noise import create_cache, create_random_noise_latents
from kdit.models.model_key import ModelKey


class TestCreateRandomNoiseLatents(unittest.TestCase):
    """create_random_noise_latents 创建随机噪声。"""

    def _make_runtime_config(self, seed=42):
        return RuntimeConfig(
            size=(512, 512),
            frame_num=16,
            seed=seed,
        )

    def test_output_shape(self):
        """输出 shape 应为 [total_samples_num, *noise_shape]。"""
        cfg = self._make_runtime_config(seed=42)
        noise, seed_g = create_random_noise_latents(
            total_samples_num=3,
            noise_shape=(4, 16, 32, 32),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(noise.shape, (3, 4, 16, 32, 32))
        self.assertIsInstance(seed_g, torch.Generator)

    def test_output_dtype(self):
        cfg = self._make_runtime_config(seed=42)
        noise, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 8, 16, 16),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )
        self.assertEqual(noise.dtype, torch.float16)

    def test_deterministic_with_same_seed(self):
        """相同 seed 应产生相同噪声。"""
        cfg = self._make_runtime_config(seed=123)
        noise1, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        noise2, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(noise1, noise2))

    def test_different_seed_produces_different_noise(self):
        cfg1 = self._make_runtime_config(seed=1)
        cfg2 = self._make_runtime_config(seed=2)
        noise1, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        noise2, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertFalse(torch.equal(noise1, noise2))

    def test_negative_seed_uses_random(self):
        """seed < 0 时应使用随机 seed（不抛异常）。"""
        cfg = self._make_runtime_config(seed=-1)
        noise, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(noise.shape, (1, 4, 4, 8, 8))

    def test_none_seed_uses_random(self):
        """seed=None 时应使用随机 seed（不抛异常）。"""
        cfg = self._make_runtime_config(seed=None)
        noise, _ = create_random_noise_latents(
            total_samples_num=1,
            noise_shape=(4, 4, 8, 8),
            runtime_config=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(noise.shape, (1, 4, 4, 8, 8))


class TestCreateCache(unittest.TestCase):
    """create_cache 根据 cache_config 创建 hybrid cache 列表。"""

    def test_none_returns_none(self):
        self.assertIsNone(create_cache(None, ModelKey.Wan2_2_T2V_14B))

    @patch("kdit.generators.steps.noise.create_hybrid_cache")
    def test_creates_cache_for_each_config(self, mock_create):
        mock_create.return_value = MagicMock(name="hybrid_cache")
        step_cache = StepCacheConfig(name="teacache")
        result = create_cache([step_cache, step_cache], ModelKey.Wan2_2_T2V_14B)
        self.assertEqual(len(result), 2)
        self.assertEqual(mock_create.call_count, 2)

    @patch("kdit.generators.steps.noise.create_hybrid_cache")
    def test_none_element_preserved(self, mock_create):
        result = create_cache([None], ModelKey.Wan2_2_T2V_14B)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0])
        mock_create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
