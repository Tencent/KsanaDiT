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

"""VaceDenoiseHandler 单元测试。

测试覆盖:
- init_denoising_loop (委托 parse_video_control_kwargs)
- get_step_kwargs (委托 get_step_video_control)
- apply_cfg (实验性 CFG / 回退到 super)
- post_noise_prediction (temporal score rescaling)
- finalize_step (bidirectional sampling 开关)
- post_run (vace trim)
- prepare_model_forward_kargs (vace_context 注入 + slg/feta)
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.generators.handlers_impl.vace_denoise import (
    VaceDenoiseHandler,
)

_PATCH_PREFIX = "kdit.generators.handlers_impl.vace_denoise"


class TestVaceInitDenoisingLoop(unittest.TestCase):
    """测试 init_denoising_loop — 委托。"""

    @patch(f"{_PATCH_PREFIX}.parse_video_control_kwargs")
    def test_delegates_to_parse(self, mock_parse):
        mock_parse.return_value = {"key": "value"}
        handler = VaceDenoiseHandler()
        dm = MagicMock()
        scheduler = MagicMock()
        result = handler.init_denoising_loop({"vc": 1}, dm, scheduler)
        mock_parse.assert_called_once()
        self.assertEqual(result, {"key": "value"})


class TestVaceGetStepKwargs(unittest.TestCase):
    """测试 get_step_kwargs — 委托。"""

    @patch(f"{_PATCH_PREFIX}.get_step_video_control")
    def test_delegates_to_get_step(self, mock_get):
        mock_get.return_value = {"step_key": 42}
        handler = VaceDenoiseHandler()
        result = handler.get_step_kwargs({"args": 1}, 0.5, 3, 20)
        mock_get.assert_called_once()
        self.assertEqual(result, {"step_key": 42})


class TestVaceApplyCfg(unittest.TestCase):
    """测试 apply_cfg — 实验性 CFG 和回退。"""

    @patch(f"{_PATCH_PREFIX}.apply_experimental_cfg")
    def test_with_exp_config(self, mock_exp_cfg):
        mock_exp_cfg.return_value = torch.tensor([1.0])
        handler = VaceDenoiseHandler()
        cond = torch.randn(2, 4)
        uncond = torch.randn(2, 4)
        exp_config = MagicMock()
        handler.apply_cfg(
            7.5,
            cond,
            uncond,
            experimental_config=exp_config,
            step_index=0,
            total_steps=20,
        )
        mock_exp_cfg.assert_called_once_with(7.5, cond, uncond, exp_config, 0)

    def test_fallback_to_super(self):
        handler = VaceDenoiseHandler()
        cond = torch.ones(2, 4)
        uncond = torch.zeros(2, 4)
        result = handler.apply_cfg(7.5, cond, uncond)
        expected = uncond + 7.5 * (cond - uncond)
        self.assertTrue(torch.allclose(result, expected))


class TestVacePostNoisePrediction(unittest.TestCase):
    """测试 post_noise_prediction — temporal score。"""

    @patch(f"{_PATCH_PREFIX}.apply_temporal_score_rescaling")
    def test_delegates(self, mock_tsr):
        mock_tsr.return_value = torch.tensor([2.0])
        handler = VaceDenoiseHandler()
        noise_pred = torch.randn(2, 4)
        noise_latent = torch.randn(2, 4)
        t = torch.tensor([500])
        vc_args = {"exp_config": MagicMock()}
        handler.post_noise_prediction(noise_pred, noise_latent, t, vc_args)
        mock_tsr.assert_called_once_with(
            noise_pred,
            noise_latent,
            t,
            vc_args["exp_config"],
        )


class TestVaceFinalizeStep(unittest.TestCase):
    """测试 finalize_step — bidirectional sampling。"""

    def test_no_bidirectional_returns_forward(self):
        handler = VaceDenoiseHandler()
        noise_latent = torch.randn(1, 16, 4, 8, 8)
        noise_forward = torch.randn(1, 16, 4, 8, 8)
        vc_args = {"bidirectional_sampling": False}
        result = handler.finalize_step(noise_latent, noise_forward, vc_args)
        self.assertTrue(torch.equal(result, noise_forward))

    @patch(f"{_PATCH_PREFIX}.apply_bidirectional_sampling")
    def test_bidirectional_delegates(self, mock_bidir):
        mock_bidir.return_value = torch.tensor([3.0])
        handler = VaceDenoiseHandler()
        noise_latent = torch.randn(1, 16, 4, 8, 8)
        noise_forward = torch.randn(1, 16, 4, 8, 8)
        vc_args = {
            "bidirectional_sampling": True,
            "exp_config": MagicMock(),
            "sample_scheduler_flipped": MagicMock(),
        }
        step_state = {
            "running_model": MagicMock(),
            "running_cfg_scale": 7.5,
            "timestep": torch.tensor([500]),
            "t": torch.tensor([500]),
            "iter_id": 0,
            "total_steps": 20,
            "current_step_percent": 0.0,
            "combine_cond_uncond": True,
            "positive": torch.randn(2, 77, 768),
            "negative": torch.randn(2, 77, 768),
            "base_latent": None,
            "step_kwargs": {},
            "sample_config": MagicMock(),
            "seed_g": MagicMock(),
        }
        handler.finalize_step(
            noise_latent,
            noise_forward,
            vc_args,
            **step_state,
        )
        mock_bidir.assert_called_once()


class TestVacePostRun(unittest.TestCase):
    """测试 post_run — vace trim。"""

    @patch(f"{_PATCH_PREFIX}.apply_vace_trim")
    def test_delegates(self, mock_trim):
        mock_trim.return_value = torch.tensor([4.0])
        handler = VaceDenoiseHandler()
        latents = torch.randn(1, 16, 4, 8, 8)
        vc_kwargs = {"trim_latent": 2}
        handler.post_run(latents, vc_kwargs)
        mock_trim.assert_called_once_with(latents, 2)

    @patch(f"{_PATCH_PREFIX}.apply_vace_trim")
    def test_no_trim(self, mock_trim):
        mock_trim.return_value = torch.randn(1, 16, 4, 8, 8)
        handler = VaceDenoiseHandler()
        latents = torch.randn(1, 16, 4, 8, 8)
        vc_kwargs = {}
        handler.post_run(latents, vc_kwargs)
        mock_trim.assert_called_once_with(latents, 0)


class TestVacePrepareModelForwardKargs(unittest.TestCase):
    """测试 prepare_model_forward_kargs — vace_context。"""

    def _make_args(
        self,
        cfg_scale=7.5,
        combine=True,
        vace_context=None,
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
            "vace_context": vace_context,
            "vace_context_scale": 1.0,
        }

    def test_combine_with_vace_context(self):
        vace_ctx = [
            torch.randn(2, 4),
            torch.randn(2, 4),
        ]
        handler = VaceDenoiseHandler()
        kwargs = self._make_args(
            cfg_scale=7.5,
            combine=True,
            vace_context=vace_ctx,
        )
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["phase"], "combine")
        self.assertIn("vace_context", result)
        # vace_context 在 combine 模式下会被 doubled
        self.assertEqual(len(result["vace_context"]), 4)

    def test_slg_feta_config_injected(self):
        handler = VaceDenoiseHandler()
        kwargs = self._make_args(cfg_scale=1.0, combine=False)
        kwargs["slg_config"] = MagicMock()
        kwargs["feta_config"] = MagicMock()
        kwargs["current_step_percent"] = 0.3
        result = handler.prepare_model_forward_kargs(**kwargs)
        self.assertIn("slg_config", result)
        self.assertIn("feta_config", result)
        self.assertEqual(result["current_step_percent"], 0.3)


if __name__ == "__main__":
    unittest.main()
