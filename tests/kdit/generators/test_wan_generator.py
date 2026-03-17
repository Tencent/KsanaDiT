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

"""WanGenerator 钩子方法单元测试。

测试覆盖:
- valid_noise_shape (I2V z_dim 覆写)
- cast_image_tensor_to (T2V 返回 None)
- _get_model_boundary (双模型 boundary 计算)
- _apply_input_latent (噪声混合 / add_noise_to_latent)
- get_running_model (单模型 / 双模型 + boundary 切换)
- get_running_cache (单缓存 / 双缓存 + boundary 切换)
- get_running_cfg_scale (单值 / 双值 + boundary 切换)
- prepare_model_forward_kargs (combine / split / no-cfg 模式)
"""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from kdit.models import ModelKey


def _make_wan_generator(model_key=ModelKey.Wan2_2_T2V_14B):
    """构造 WanGenerator 实例，绕过 AdvancedFactory 注册。"""
    from kdit.generators.wan_generator import WanGenerator

    gen = WanGenerator.__new__(WanGenerator)
    gen.__init__()
    gen.model_key = model_key
    return gen


def _make_diffusion_model(*, z_dim=None, num_train_timesteps=1000, boundary=None):
    """构造 mock DiffusionModel。"""
    model = MagicMock()

    @dataclass
    class _VAE:
        z_dim: int = 16

    @dataclass
    class _SampleConfig:
        num_train_timesteps: int = 1000

    @dataclass
    class _RuntimeConfig:
        boundary: float = None

    @dataclass
    class _Settings:
        vae: _VAE = None
        sample_config: _SampleConfig = None
        runtime_config: _RuntimeConfig = None

    vae = _VAE(z_dim=z_dim) if z_dim is not None else _VAE()
    sc = _SampleConfig(num_train_timesteps=num_train_timesteps)
    rc = _RuntimeConfig(boundary=boundary)
    settings = _Settings(vae=vae, sample_config=sc, runtime_config=rc)
    model.default_settings = settings
    model.model_config = MagicMock()
    model.model_config.boundary = None
    model.device = "cpu"
    model.to = MagicMock()
    return model


class TestWanValidNoiseShape(unittest.TestCase):
    """测试 valid_noise_shape — I2V 模式下覆写 z_dim。"""

    def test_t2v_keeps_original_shape(self):
        gen = _make_wan_generator(ModelKey.Wan2_2_T2V_14B)
        dm = _make_diffusion_model()
        result = gen.valid_noise_shape([16, 8, 32, 32], [dm])
        self.assertEqual(result, [16, 8, 32, 32])

    def test_i2v_overrides_z_dim(self):
        gen = _make_wan_generator(ModelKey.Wan2_2_I2V_14B)
        dm = _make_diffusion_model(z_dim=32)
        result = gen.valid_noise_shape([16, 8, 32, 32], [dm])
        self.assertEqual(result[0], 32)

    def test_i2v_missing_z_dim_raises(self):
        gen = _make_wan_generator(ModelKey.Wan2_2_I2V_14B)
        dm = _make_diffusion_model()
        # 用一个没有 z_dim 属性的空对象替换 vae
        dm.default_settings.vae = type("_EmptyVAE", (), {})()
        with self.assertRaises(ValueError, msg="vae.z_dim not found"):
            gen.valid_noise_shape([16, 8, 32, 32], [dm])


class TestWanCastImageTensor(unittest.TestCase):
    """测试 cast_image_tensor_to — T2V 返回 None。"""

    def test_t2v_returns_none(self):
        gen = _make_wan_generator(ModelKey.Wan2_2_T2V_14B)
        result = gen.cast_image_tensor_to([torch.randn(1, 4)], dtype=torch.float16, device=torch.device("cpu"))
        self.assertIsNone(result)

    def test_i2v_casts_tensors(self):
        gen = _make_wan_generator(ModelKey.Wan2_2_I2V_14B)
        embeds = [torch.randn(1, 4, dtype=torch.float32)]
        result = gen.cast_image_tensor_to(embeds, dtype=torch.float16, device=torch.device("cpu"))
        self.assertIsNotNone(result)
        self.assertEqual(result[0].dtype, torch.float16)


class TestWanGetModelBoundary(unittest.TestCase):
    """测试 _get_model_boundary — 双模型 boundary 计算。"""

    def test_single_model_returns_none(self):
        gen = _make_wan_generator()
        dm = _make_diffusion_model()
        result = gen._get_model_boundary([dm])
        self.assertIsNone(result)

    def test_dual_model_with_boundary(self):
        gen = _make_wan_generator()
        high = _make_diffusion_model(num_train_timesteps=1000, boundary=0.5)
        high.model_config.boundary = 0.5
        low = MagicMock()
        result = gen._get_model_boundary([high, low])
        self.assertEqual(result, 500.0)  # 0.5 * 1000

    def test_dual_model_no_boundary_raises(self):
        gen = _make_wan_generator()
        high = _make_diffusion_model()
        high.model_config.boundary = None
        low = MagicMock()
        with self.assertRaises(RuntimeError, msg="boundary should be set"):
            gen._get_model_boundary([high, low])

    def test_cached_boundary(self):
        gen = _make_wan_generator()
        gen.boundary = 42.0
        result = gen._get_model_boundary([MagicMock()])
        self.assertEqual(result, 42.0)


class TestWanApplyInputLatent(unittest.TestCase):
    """测试 _apply_input_latent — 噪声混合逻辑。"""

    def _make_sample_config(self, add_noise=True):
        sc = MagicMock()
        sc.add_noise_to_latent = add_noise
        return sc

    def test_none_input_returns_noise(self):
        gen = _make_wan_generator()
        noise = torch.randn(1, 16, 4, 8, 8)
        result = gen._apply_input_latent(noise, None, self._make_sample_config(), torch.tensor([500]), 1000)
        self.assertTrue(torch.equal(result, noise))

    def test_non_5d_raises(self):
        gen = _make_wan_generator()
        noise = torch.randn(1, 16, 8, 8)  # 4D
        with self.assertRaises(ValueError, msg="must be 5D"):
            gen._apply_input_latent(
                noise, torch.randn(1, 16, 8, 8), self._make_sample_config(), torch.tensor([500]), 1000
            )

    def test_add_noise_blending(self):
        gen = _make_wan_generator()
        noise = torch.ones(1, 16, 4, 8, 8)
        input_lat = torch.zeros(1, 16, 4, 8, 8)
        timesteps = torch.tensor([500])
        result = gen._apply_input_latent(noise, input_lat, self._make_sample_config(True), timesteps, 1000)
        # result = noise * (t/T) + (1 - t/T) * input = 1 * 0.5 + 0 * 0.5 = 0.5
        self.assertTrue(torch.allclose(result, torch.full_like(result, 0.5)))

    def test_no_noise_uses_input_directly(self):
        gen = _make_wan_generator()
        noise = torch.ones(1, 16, 4, 8, 8)
        input_lat = torch.full((1, 16, 4, 8, 8), 2.0)
        result = gen._apply_input_latent(noise, input_lat, self._make_sample_config(False), torch.tensor([500]), 1000)
        self.assertTrue(torch.equal(result, input_lat))

    def test_frame_padding(self):
        gen = _make_wan_generator()
        noise = torch.randn(1, 16, 8, 8, 8)  # 8 frames
        input_lat = torch.randn(1, 16, 4, 8, 8)  # 4 frames
        result = gen._apply_input_latent(noise, input_lat, self._make_sample_config(False), torch.tensor([500]), 1000)
        self.assertEqual(result.shape, noise.shape)


class TestWanGetRunningModel(unittest.TestCase):
    """测试 get_running_model — 单/双模型切换。"""

    def test_single_model(self):
        gen = _make_wan_generator()
        dm = MagicMock()
        result = gen.get_running_model([dm], timestep_id=500, device="cuda:0")
        self.assertIs(result, dm)

    def test_dual_model_high(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        high = MagicMock()
        high.device = "cuda:0"
        low = MagicMock()
        low.device = "cuda:0"
        result = gen.get_running_model([high, low], timestep_id=600, device="cuda:0", offload_device="cpu")
        self.assertIs(result, high)

    def test_dual_model_low(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        high = MagicMock()
        high.device = "cuda:0"
        low = MagicMock()
        low.device = "cuda:0"
        result = gen.get_running_model([high, low], timestep_id=400, device="cuda:0", offload_device="cpu")
        self.assertIs(result, low)

    def test_missing_device_raises(self):
        gen = _make_wan_generator()
        with self.assertRaises(ValueError, msg="device must be provided"):
            gen.get_running_model([MagicMock()], timestep_id=0, device=None)


class TestWanGetRunningCache(unittest.TestCase):
    """测试 get_running_cache — 单/双缓存切换。"""

    def test_non_list_passthrough(self):
        gen = _make_wan_generator()
        cache = MagicMock()
        result = gen.get_running_cache(cache, timestep_id=0)
        self.assertIs(result, cache)

    def test_single_cache(self):
        gen = _make_wan_generator()
        cache = MagicMock()
        result = gen.get_running_cache([cache], timestep_id=0)
        self.assertIs(result, cache)

    def test_dual_cache_high(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        high_cache = MagicMock()
        low_cache = MagicMock()
        result = gen.get_running_cache([high_cache, low_cache], timestep_id=600)
        self.assertIs(result, high_cache)

    def test_dual_cache_low(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        high_cache = MagicMock()
        low_cache = MagicMock()
        result = gen.get_running_cache([high_cache, low_cache], timestep_id=400)
        self.assertIs(result, low_cache)


class TestWanGetRunningCfgScale(unittest.TestCase):
    """测试 get_running_cfg_scale — 单/双 cfg_scale 切换。"""

    def test_scalar(self):
        gen = _make_wan_generator()
        self.assertEqual(gen.get_running_cfg_scale(7.5, timestep_id=0), 7.5)

    def test_single_list(self):
        gen = _make_wan_generator()
        self.assertEqual(gen.get_running_cfg_scale([7.5], timestep_id=0), 7.5)

    def test_dual_above_boundary(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        self.assertEqual(gen.get_running_cfg_scale([7.5, 3.0], timestep_id=600), 7.5)

    def test_dual_below_boundary(self):
        gen = _make_wan_generator()
        gen.boundary = 500
        self.assertEqual(gen.get_running_cfg_scale([7.5, 3.0], timestep_id=400), 3.0)


class TestWanPrepareModelForwardKargs(unittest.TestCase):
    """测试 prepare_model_forward_kargs — combine/split/no-cfg 模式。"""

    def _make_args(self, *, cfg_scale=7.5, combine=True, model_key=ModelKey.Wan2_2_T2V_14B):
        gen = _make_wan_generator(model_key)
        noise = torch.randn(2, 16, 4, 8, 8)
        timestep = torch.tensor([500])
        pos = torch.randn(2, 77, 768)
        neg = torch.randn(2, 77, 768)
        return gen, {
            "cfg_scale": cfg_scale,
            "noise_latent": noise,
            "timestep": timestep,
            "combine_cond_uncond": combine,
            "step_iter": 0,
            "cache": None,
            "positive": pos,
            "negative": neg,
            "image_embeds": None,
        }

    def test_combine_mode(self):
        gen, kwargs = self._make_args(cfg_scale=7.5, combine=True)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "combine")
        self.assertEqual(result["x"].shape[0], 4)  # 2 * 2

    def test_split_mode(self):
        gen, kwargs = self._make_args(cfg_scale=7.5, combine=False)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["phase"], "cond")
        self.assertEqual(result[1]["phase"], "uncond")

    def test_no_cfg(self):
        gen, kwargs = self._make_args(cfg_scale=1.0, combine=False)
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "cond")

    def test_i2v_includes_y(self):
        gen, kwargs = self._make_args(cfg_scale=7.5, combine=True, model_key=ModelKey.Wan2_2_I2V_14B)
        kwargs["image_embeds"] = [torch.randn(2, 256)]
        result = gen.prepare_model_forward_kargs(**kwargs)
        self.assertIn("y", result)
        self.assertEqual(result["y"].shape[0], 4)


if __name__ == "__main__":
    unittest.main()
