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

"""WanDenoiseHandler 单元测试。

测试覆盖:
- _get_model_boundary (单/双模型 boundary 计算)
- get_running_model (单/双模型 + boundary 切换)
- get_running_cache (单/双缓存 + boundary 切换)
- get_running_cfg_scale (单/双 cfg_scale + boundary 切换)
- prepare_model_forward_kargs (combine / split / no-cfg 模式)
"""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from kdit.generators.handlers_impl.wan_denoise import WanDenoiseHandler
from kdit.models.model_key import ModelKey


def _make_diffusion_model(*, num_train_timesteps=1000, boundary=None):
    """构造 mock DiffusionModel。"""
    model = MagicMock()

    @dataclass
    class _SampleConfig:
        num_train_timesteps: int = 1000

    @dataclass
    class _RuntimeConfig:
        boundary: float = None

    @dataclass
    class _Settings:
        sample_config: _SampleConfig = None
        runtime_config: _RuntimeConfig = None

    sc = _SampleConfig(num_train_timesteps=num_train_timesteps)
    rc = _RuntimeConfig(boundary=boundary)
    settings = _Settings(sample_config=sc, runtime_config=rc)
    model.default_settings = settings
    model.model_config = MagicMock()
    model.model_config.boundary = None
    model.device = "cpu"
    model.to = MagicMock()
    return model


class TestWanGetModelBoundary(unittest.TestCase):
    """测试 _get_model_boundary — 双模型 boundary 计算。"""

    def test_single_model_returns_none(self):
        handler = WanDenoiseHandler()
        dm = _make_diffusion_model()
        result = handler._get_model_boundary([dm])
        self.assertIsNone(result)

    def test_dual_model_with_boundary(self):
        handler = WanDenoiseHandler()
        high = _make_diffusion_model(num_train_timesteps=1000, boundary=0.5)
        high.model_config.boundary = 0.5
        low = MagicMock()
        result = handler._get_model_boundary([high, low])
        self.assertEqual(result, 500.0)  # 0.5 * 1000

    def test_dual_model_no_boundary_raises(self):
        handler = WanDenoiseHandler()
        high = _make_diffusion_model()
        high.model_config.boundary = None
        low = MagicMock()
        with self.assertRaises(RuntimeError):
            handler._get_model_boundary([high, low])

    def test_cached_boundary(self):
        handler = WanDenoiseHandler()
        handler._boundary = 42.0
        result = handler._get_model_boundary([MagicMock()])
        self.assertEqual(result, 42.0)


class TestWanGetRunningModel(unittest.TestCase):
    """测试 get_running_model — 单/双模型切换。"""

    def test_single_model(self):
        handler = WanDenoiseHandler()
        model = MagicMock()
        model.device = "cpu"
        result = handler.get_running_model(
            [model],
            timestep_id=500,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        self.assertIs(result, model)

    def test_dual_model_high(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        high = MagicMock(device="cpu")
        low = MagicMock(device="cpu")
        result = handler.get_running_model(
            [high, low],
            timestep_id=600,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        self.assertIs(result, high)

    def test_dual_model_low(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        high = MagicMock(device="cpu")
        low = MagicMock(device="cpu")
        result = handler.get_running_model(
            [high, low],
            timestep_id=400,
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        self.assertIs(result, low)

    def test_missing_device_raises(self):
        handler = WanDenoiseHandler()
        with self.assertRaises(ValueError):
            handler.get_running_model(
                [MagicMock()],
                timestep_id=0,
                device=None,
                offload_device=torch.device("cpu"),
            )


class TestWanGetRunningCache(unittest.TestCase):
    """测试 get_running_cache — 单/双缓存切换。"""

    def test_non_list_passthrough(self):
        handler = WanDenoiseHandler()
        cache = MagicMock()
        result = handler.get_running_cache(cache, timestep_id=0)
        self.assertIs(result, cache)

    def test_single_cache(self):
        handler = WanDenoiseHandler()
        cache = MagicMock()
        result = handler.get_running_cache([cache], timestep_id=0)
        self.assertIs(result, cache)

    def test_dual_cache_high(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        high_cache = MagicMock()
        low_cache = MagicMock()
        result = handler.get_running_cache([high_cache, low_cache], timestep_id=600)
        self.assertIs(result, high_cache)

    def test_dual_cache_low(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        high_cache = MagicMock()
        low_cache = MagicMock()
        result = handler.get_running_cache([high_cache, low_cache], timestep_id=400)
        self.assertIs(result, low_cache)


class TestWanGetRunningCfgScale(unittest.TestCase):
    """测试 get_running_cfg_scale — 单/双 cfg_scale 切换。"""

    def test_dual_above_boundary(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        result = handler.get_running_cfg_scale([7.5, 3.0], timestep_id=600)
        self.assertEqual(result, 7.5)

    def test_dual_below_boundary(self):
        handler = WanDenoiseHandler()
        handler._boundary = 500
        result = handler.get_running_cfg_scale([7.5, 3.0], timestep_id=400)
        self.assertEqual(result, 3.0)


class TestWanPrepareModelForwardKargs(unittest.TestCase):
    """测试 prepare_model_forward_kargs — combine/split/no-cfg。"""

    def _make_args(
        self,
        *,
        cfg_scale=7.5,
        combine=True,
        model_key=ModelKey.Wan2_2_T2V_14B,
    ):
        noise = torch.randn(2, 16, 4, 8, 8)
        timestep = torch.tensor([500])
        pos = torch.randn(2, 77, 768)
        neg = torch.randn(2, 77, 768)
        return {
            "cfg_scale": cfg_scale,
            "noise_latent": noise,
            "timestep": timestep,
            "combine_cond_uncond": combine,
            "step_iter": 0,
            "cache": None,
            "positive": pos,
            "negative": neg,
            "base_latent": None,
            "model_key": model_key,
        }

    def test_combine_mode(self):
        handler = WanDenoiseHandler()
        kwargs = self._make_args(cfg_scale=7.5, combine=True)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "combine")
        self.assertEqual(result["x"].shape[0], 4)  # 2 * 2

    def test_split_mode(self):
        handler = WanDenoiseHandler()
        kwargs = self._make_args(cfg_scale=7.5, combine=False)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["phase"], "cond")
        self.assertEqual(result[1]["phase"], "uncond")

    def test_no_cfg(self):
        handler = WanDenoiseHandler()
        kwargs = self._make_args(cfg_scale=1.0, combine=False)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "cond")

    def test_i2v_includes_y(self):
        handler = WanDenoiseHandler()
        kwargs = self._make_args(
            cfg_scale=7.5,
            combine=True,
            model_key=ModelKey.Wan2_2_I2V_14B,
        )
        kwargs["base_latent"] = torch.randn(2, 256)
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIn("y", result)
        self.assertEqual(result["y"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
