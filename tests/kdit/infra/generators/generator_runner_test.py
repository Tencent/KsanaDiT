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

"""GeneratorRunner 单元测试 — mock 所有 Handler 测试主流程编排。

测试覆盖:
- __init__ 构造
- run() 主流程中 Handler 方法调用顺序
- _use_cfg() 辅助方法
- _get_num_train_timesteps()
- _get_patch_size()
"""

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import torch

from kdit.config import RuntimeConfig, SampleConfig, SolverType
from kdit.generators.generator_context import BaseLatent, GeneratorInferContext
from kdit.generators.generator_def import GeneratorDef
from kdit.generators.generator_runner import GeneratorRunner
from kdit.generators.handlers.denoise_handler import DenoiseHandler
from kdit.generators.handlers.latent_handler import LatentHandler
from kdit.generators.handlers.text_handler import TextHandler
from kdit.models.model_key import ModelKey

# ── Mock 构造辅助函数 ──────────────────────────────────────────────────


def _make_mock_context(
    *,
    num_prompts=1,
    batch_size_per_prompts=None,
    has_aux=False,
):
    """构造 mock GeneratorInferContext。"""
    positive = torch.randn(num_prompts, 77, 768)
    negative = torch.randn(num_prompts, 77, 768)
    base_latent = BaseLatent(latent=torch.randn(num_prompts, 16, 4, 8, 8))
    aux_latent = None
    if has_aux:
        from kdit.generators.generator_context import AuxLatent

        aux_latent = AuxLatent(latent=torch.randn(num_prompts, 16, 4, 8, 8))

    if batch_size_per_prompts is None:
        batch_size_per_prompts = [1] * num_prompts

    sample_config = SampleConfig(steps=2, cfg_scale=7.5, solver=SolverType.EULER)
    runtime_config = RuntimeConfig(
        size=(64, 64),
        frame_num=4,
        seed=42,
        batch_size_per_prompts=batch_size_per_prompts,
    )

    return GeneratorInferContext(
        diffusion_model=MagicMock(),
        positive=positive,
        negative=negative,
        base_latent=base_latent,
        aux_latent=aux_latent,
        device=torch.device("cpu"),
        sample_config=sample_config,
        runtime_config=runtime_config,
    )


def _make_mock_diffusion_model(
    *,
    model_key=ModelKey.Wan2_2_T2V_14B,
    num_train_timesteps=1000,
    patch_size=2,
):
    """构造 mock DiffusionModel。"""
    model = MagicMock()

    @dataclass
    class _SampleConfig:
        num_train_timesteps: int = 1000

    @dataclass
    class _DiffusionConfig:
        patch_size: int = 2

    @dataclass
    class _Settings:
        sample_config: _SampleConfig = None
        diffusion: _DiffusionConfig = None

    sc = _SampleConfig(num_train_timesteps=num_train_timesteps)
    dc = _DiffusionConfig(patch_size=patch_size)
    settings = _Settings(sample_config=sc, diffusion=dc)
    model.default_settings = settings
    model.run_dtype = torch.float16
    model.model_key = model_key
    model.device = torch.device("cpu")
    model.to = MagicMock()
    model.forward = MagicMock(return_value=torch.randn(1, 16, 4, 8, 8))
    return model


def _make_mock_generator_def():
    """构造 mock GeneratorDef，所有 Handler 使用 MagicMock。"""
    text_h = MagicMock(spec=TextHandler)
    text_h.preprocess.side_effect = lambda x: x
    text_h.get_num_prompts.return_value = 1
    text_h.validate.side_effect = lambda p, n: (p, n)
    text_h.expand_to_batch.side_effect = lambda p, n, b: (p, n)
    text_h.cast_to.side_effect = lambda p, n, **kw: (p, n)
    text_h._expand_to_total_prompts_size.side_effect = lambda t, b: t

    latent_h = MagicMock(spec=LatentHandler)
    latent_h.preprocess_base.side_effect = lambda x: x[0]
    latent_h.validate_noise_shape.side_effect = lambda s, m, mk: s
    latent_h.pack_noise.side_effect = lambda x, p: x
    latent_h.unpack_noise.side_effect = lambda x, p: x
    latent_h.pack_aux.side_effect = lambda x, p: x
    latent_h.maybe_update_sample_config.side_effect = lambda s, *a: s
    latent_h.apply_aux_latent.side_effect = lambda nl, al, sc, ts, ntt: nl

    denoise_h = MagicMock(spec=DenoiseHandler)
    denoise_h.build_video_control_kwargs.return_value = {}
    denoise_h.post_run.side_effect = lambda x, v: x
    denoise_h.init_denoising_loop.return_value = {}
    denoise_h.get_running_model.return_value = MagicMock(
        device=torch.device("cpu"),
        forward=MagicMock(return_value=torch.randn(2, 16, 4, 8, 8)),
    )
    denoise_h.get_running_cache.return_value = None
    denoise_h.get_running_cfg_scale.return_value = 7.5
    denoise_h.get_step_kwargs.return_value = {}
    denoise_h.prepare_model_forward_kargs.return_value = {"x": torch.randn(2, 16, 4, 8, 8)}
    denoise_h.apply_cfg.side_effect = lambda c, cond, uncond, **kw: cond
    denoise_h.post_noise_prediction.side_effect = lambda p, *a: p
    denoise_h.finalize_step.side_effect = lambda n, nf, dv, **kw: nf

    gdef = GeneratorDef(
        model_key=ModelKey.Wan2_2_T2V_14B,
        text_handler=text_h,
        latent_handler=latent_h,
        denoise_handler=denoise_h,
    )
    return gdef, text_h, latent_h, denoise_h


# ── 测试类 ─────────────────────────────────────────────────────────────


class TestGeneratorRunnerInit(unittest.TestCase):
    """测试 GeneratorRunner 构造。"""

    def test_init_with_generator_def(self):
        """验证 handler 属性正确设置。"""
        gdef, text_h, latent_h, denoise_h = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        self.assertIs(runner.gdef, gdef)
        self.assertEqual(runner.model_key, ModelKey.Wan2_2_T2V_14B)
        self.assertIs(runner._text, text_h)
        self.assertIs(runner._latent, latent_h)
        self.assertIs(runner._denoise, denoise_h)

    def test_init_creates_batch_scheduler(self):
        """验证构造时创建了 batch_scheduler。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)
        self.assertIsNotNone(runner.batch_scheduler)


# 需要 mock 的模块路径前缀
_MOD = "kdit.generators.generator_runner"


class TestGeneratorRunnerRun(unittest.TestCase):
    """测试 run() 主流程 — mock 所有外部依赖。"""

    def _make_runner_and_ctx(self):
        """构造 runner + ctx + mock model 的完整测试环境。"""
        gdef, text_h, latent_h, denoise_h = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = _make_mock_diffusion_model()
        ctx = _make_mock_context()
        ctx.diffusion_model = mock_model

        return runner, ctx, text_h, latent_h, denoise_h, mock_model

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_calls_text_handler_methods(self, mock_tensor_ops, mock_valid, mock_noise, mock_sched):
        """验证 text handler 的 preprocess/validate/expand/cast 被调用。"""
        runner, ctx, text_h, _, _, mock_model = self._make_runner_and_ctx()

        mock_valid.valid_diffusion_model.return_value = [mock_model]
        mock_valid.valid_sample_config.side_effect = lambda s, n: s
        mock_valid.valid_cache_config.side_effect = lambda c, n: c
        mock_valid.valid_runtime_config.side_effect = lambda r, n: r
        mock_valid.valid_aux_latent.return_value = None
        mock_tensor_ops.cast_to.side_effect = lambda t, **kw: t
        mock_tensor_ops.split_tensors.side_effect = lambda t, s, e: t

        noise_t = torch.randn(1, 16, 4, 8, 8)
        mock_noise.create_random_noise_latents.return_value = (
            noise_t,
            MagicMock(),
        )
        mock_noise.create_cache.return_value = [None]

        mock_sched.return_value = (
            MagicMock(step=MagicMock(return_value=(noise_t, None))),
            None,
            torch.tensor([0.9, 0.1]),
        )

        runner.batch_scheduler = MagicMock()
        runner.batch_scheduler.build_batch_strategy.return_value = [MagicMock(start=0, end=1, combine_cond_uncond=True)]

        runner.run(ctx)

        # text handler 方法应被调用
        text_h.preprocess.assert_called()
        text_h.get_num_prompts.assert_called()
        text_h.validate.assert_called()
        text_h.expand_to_batch.assert_called()
        text_h.cast_to.assert_called()

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_calls_latent_handler_methods(self, mock_tensor_ops, mock_valid, mock_noise, mock_sched):
        """验证 latent handler 的 preprocess_base/validate_noise_shape/pack_noise 被调用。"""
        runner, ctx, _, latent_h, _, mock_model = self._make_runner_and_ctx()

        mock_valid.valid_diffusion_model.return_value = [mock_model]
        mock_valid.valid_sample_config.side_effect = lambda s, n: s
        mock_valid.valid_cache_config.side_effect = lambda c, n: c
        mock_valid.valid_runtime_config.side_effect = lambda r, n: r
        mock_valid.valid_aux_latent.return_value = None
        mock_tensor_ops.cast_to.side_effect = lambda t, **kw: t
        mock_tensor_ops.split_tensors.side_effect = lambda t, s, e: t

        noise_t = torch.randn(1, 16, 4, 8, 8)
        mock_noise.create_random_noise_latents.return_value = (
            noise_t,
            MagicMock(),
        )
        mock_noise.create_cache.return_value = [None]

        mock_sched.return_value = (
            MagicMock(step=MagicMock(return_value=(noise_t, None))),
            None,
            torch.tensor([0.9, 0.1]),
        )

        runner.batch_scheduler = MagicMock()
        runner.batch_scheduler.build_batch_strategy.return_value = [MagicMock(start=0, end=1, combine_cond_uncond=True)]

        runner.run(ctx)

        latent_h.preprocess_base.assert_called()
        latent_h.validate_noise_shape.assert_called()
        latent_h.pack_noise.assert_called()
        latent_h.unpack_noise.assert_called()

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_calls_denoise_handler_methods(self, mock_tensor_ops, mock_valid, mock_noise, mock_sched):
        """验证 denoise handler 的 build_video_control_kwargs/post_run 被调用。"""
        runner, ctx, _, _, denoise_h, mock_model = self._make_runner_and_ctx()

        mock_valid.valid_diffusion_model.return_value = [mock_model]
        mock_valid.valid_sample_config.side_effect = lambda s, n: s
        mock_valid.valid_cache_config.side_effect = lambda c, n: c
        mock_valid.valid_runtime_config.side_effect = lambda r, n: r
        mock_valid.valid_aux_latent.return_value = None
        mock_tensor_ops.cast_to.side_effect = lambda t, **kw: t
        mock_tensor_ops.split_tensors.side_effect = lambda t, s, e: t

        noise_t = torch.randn(1, 16, 4, 8, 8)
        mock_noise.create_random_noise_latents.return_value = (
            noise_t,
            MagicMock(),
        )
        mock_noise.create_cache.return_value = [None]

        mock_sched.return_value = (
            MagicMock(step=MagicMock(return_value=(noise_t, None))),
            None,
            torch.tensor([0.9, 0.1]),
        )

        runner.batch_scheduler = MagicMock()
        runner.batch_scheduler.build_batch_strategy.return_value = [MagicMock(start=0, end=1, combine_cond_uncond=True)]

        runner.run(ctx)

        denoise_h.build_video_control_kwargs.assert_called()
        denoise_h.post_run.assert_called()

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_passes_model_key_to_prepare_model_forward_kargs(
        self, mock_tensor_ops, mock_valid, mock_noise, mock_sched
    ):
        """验证 _run_one_batch 调用 prepare_model_forward_kargs 时传递了 model_key。

        回归测试：修复前 model_key 未传递，导致 WanDenoiseHandler 因缺少
        必需参数 model_key 而 TypeError，I2V 的 y 无法传给模型。
        """
        runner, ctx, _, _, denoise_h, mock_model = self._make_runner_and_ctx()

        mock_valid.valid_diffusion_model.return_value = [mock_model]
        mock_valid.valid_sample_config.side_effect = lambda s, n: s
        mock_valid.valid_cache_config.side_effect = lambda c, n: c
        mock_valid.valid_runtime_config.side_effect = lambda r, n: r
        mock_valid.valid_aux_latent.return_value = None
        mock_tensor_ops.cast_to.side_effect = lambda t, **kw: t
        mock_tensor_ops.split_tensors.side_effect = lambda t, s, e: t

        noise_t = torch.randn(1, 16, 4, 8, 8)
        mock_noise.create_random_noise_latents.return_value = (noise_t, MagicMock())
        mock_noise.create_cache.return_value = [None]

        mock_sched.return_value = (
            MagicMock(step=MagicMock(return_value=(noise_t, None))),
            None,
            torch.tensor([0.9, 0.1]),
        )

        runner.batch_scheduler = MagicMock()
        runner.batch_scheduler.build_batch_strategy.return_value = [MagicMock(start=0, end=1, combine_cond_uncond=True)]

        runner.run(ctx)

        # 验证 prepare_model_forward_kargs 被调用时包含 model_key 关键字参数
        denoise_h.prepare_model_forward_kargs.assert_called()
        call_kwargs = denoise_h.prepare_model_forward_kargs.call_args
        self.assertIn("model_key", call_kwargs.kwargs, "model_key must be passed to prepare_model_forward_kargs")
        self.assertEqual(call_kwargs.kwargs["model_key"], runner.model_key)

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_returns_tensor(self, mock_tensor_ops, mock_valid, mock_noise, mock_sched):
        """验证 run() 返回值是 tensor。"""
        runner, ctx, _, _, _, mock_model = self._make_runner_and_ctx()

        mock_valid.valid_diffusion_model.return_value = [mock_model]
        mock_valid.valid_sample_config.side_effect = lambda s, n: s
        mock_valid.valid_cache_config.side_effect = lambda c, n: c
        mock_valid.valid_runtime_config.side_effect = lambda r, n: r
        mock_valid.valid_aux_latent.return_value = None
        mock_tensor_ops.cast_to.side_effect = lambda t, **kw: t
        mock_tensor_ops.split_tensors.side_effect = lambda t, s, e: t

        noise_t = torch.randn(1, 16, 4, 8, 8)
        mock_noise.create_random_noise_latents.return_value = (
            noise_t,
            MagicMock(),
        )
        mock_noise.create_cache.return_value = [None]

        mock_sched.return_value = (
            MagicMock(step=MagicMock(return_value=(noise_t, None))),
            None,
            torch.tensor([0.9, 0.1]),
        )

        runner.batch_scheduler = MagicMock()
        runner.batch_scheduler.build_batch_strategy.return_value = [MagicMock(start=0, end=1, combine_cond_uncond=True)]

        result = runner.run(ctx)
        self.assertIsInstance(result, torch.Tensor)

    @patch(f"{_MOD}.get_sample_scheduler")
    @patch(f"{_MOD}.noise_ops")
    @patch(f"{_MOD}.validation")
    @patch(f"{_MOD}.tensor_ops")
    def test_run_raises_without_base_latent(self, mock_tensor_ops, mock_valid, mock_noise, mock_sched):
        """base_latent 为 None 时应 raise ValueError。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        ctx = _make_mock_context()
        ctx.base_latent = None

        with self.assertRaises(ValueError):
            runner.run(ctx)


class TestGeneratorRunnerUseCfg(unittest.TestCase):
    """测试 _use_cfg() 静态辅助方法。"""

    def test_cfg_scale_above_one(self):
        """cfg_scale > 1 返回 True。"""
        self.assertTrue(GeneratorRunner._use_cfg(7.5))

    def test_cfg_scale_one(self):
        """cfg_scale == 1 返回 False。"""
        self.assertFalse(GeneratorRunner._use_cfg(1.0))

    def test_cfg_scale_below_one(self):
        """cfg_scale < 1 但距离 1.0 > eps 时返回 True（需要 CFG）。"""
        self.assertTrue(GeneratorRunner._use_cfg(0.5))

    def test_cfg_scale_slightly_above_one(self):
        """cfg_scale 略大于 1 (1.0 + 1e-7) 仍返回 False（在 eps 内）。"""
        self.assertFalse(GeneratorRunner._use_cfg(1.0 + 1e-7))

    def test_cfg_scale_clearly_above_one(self):
        """cfg_scale = 1.001 返回 True（超出 eps）。"""
        self.assertTrue(GeneratorRunner._use_cfg(1.001))

    def test_cfg_scale_zero(self):
        """cfg_scale = 0 返回 True（距离 1.0 > eps）。"""
        self.assertTrue(GeneratorRunner._use_cfg(0.0))


class TestGeneratorRunnerGetNumTrainTimesteps(unittest.TestCase):
    """测试 _get_num_train_timesteps()。"""

    def test_returns_from_settings(self):
        """从 default_settings.sample_config 获取 num_train_timesteps。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = _make_mock_diffusion_model(num_train_timesteps=1000)
        result = runner._get_num_train_timesteps(mock_model.default_settings)
        self.assertEqual(result, 1000)

    def test_custom_value(self):
        """自定义 num_train_timesteps 值。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = _make_mock_diffusion_model(num_train_timesteps=500)
        result = runner._get_num_train_timesteps(mock_model.default_settings)
        self.assertEqual(result, 500)

    def test_missing_raises(self):
        """num_train_timesteps 为 None 时应 raise RuntimeError。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        @dataclass
        class _SampleConfig:
            num_train_timesteps: int = None

        @dataclass
        class _Settings:
            sample_config: _SampleConfig = None

        settings = _Settings(sample_config=_SampleConfig())
        with self.assertRaises(RuntimeError):
            runner._get_num_train_timesteps(settings)


class TestGeneratorRunnerGetPatchSize(unittest.TestCase):
    """测试 _get_patch_size()。"""

    def test_returns_patch_size_from_settings(self):
        """从 default_settings.diffusion.patch_size 获取。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = _make_mock_diffusion_model(patch_size=2)
        # diffusion_model 本身没有 patch_size 属性
        del mock_model.patch_size
        result = runner._get_patch_size([mock_model])
        self.assertEqual(result, 2)

    def test_returns_patch_size_from_model_attr(self):
        """从 diffusion_model.patch_size 属性获取（优先级更高）。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = _make_mock_diffusion_model(patch_size=2)
        # 模型列表本身有 patch_size 属性 — 不使用 wraps 避免 list 属性穿透
        model_list_mock = MagicMock()
        model_list_mock.__getitem__ = MagicMock(side_effect=lambda i: mock_model)
        model_list_mock.__len__ = MagicMock(return_value=1)
        model_list_mock.patch_size = 4
        result = runner._get_patch_size(model_list_mock)
        self.assertEqual(result, 4)

    def test_no_patch_size_raises(self):
        """无 patch_size 属性时应 raise RuntimeError。"""
        gdef, _, _, _ = _make_mock_generator_def()
        runner = GeneratorRunner(gdef)

        mock_model = MagicMock()

        @dataclass
        class _DiffusionConfig:
            pass  # no patch_size

        @dataclass
        class _Settings:
            diffusion: _DiffusionConfig = None

        mock_model.default_settings = _Settings(diffusion=_DiffusionConfig())
        # 确保 diffusion_model 列表也没有 patch_size
        model_list = [mock_model]

        with self.assertRaises(RuntimeError):
            runner._get_patch_size(model_list)


if __name__ == "__main__":
    unittest.main()
