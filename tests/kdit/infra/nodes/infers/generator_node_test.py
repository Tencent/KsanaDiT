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

"""Tests for kdit.nodes.infers.generator_node — GeneratorNode.run()。

使用 mock 替代 model_pool / tensor_pool / GeneratorRunner，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.config import RuntimeConfig, SampleConfig, SolverType
from kdit.generators.generator_context import GeneratorInferContext
from kdit.models.model_key import ModelKey
from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import NodeDispatchPolicy
from kdit.nodes.infers.generator_node import GeneratorNode
from kdit.tensor import TensorKey
from kdit.tensor.tensor_value import TensorValue


class TestGeneratorNode(unittest.TestCase):
    """GeneratorNode 从 tensor_pool 读取数据，构造 GeneratorInferContext 并调用 generator.run()。"""

    def setUp(self):
        self.node = GeneratorNode()

        # mock tensor_pool
        self.tensor_pool = MagicMock()
        self.positive = torch.randn(1, 77, 768)
        self.negative = torch.randn(1, 77, 768)
        self.base_latent = [torch.randn(1, 4, 16, 32, 32)]
        self.aux_latent = torch.randn(1, 4, 16, 32, 32)

        def _get_side_effect(key):
            """返回 TensorValue 包装，与 _get_data() 中 v.data 配合。"""
            mapping = {
                TensorKey.POSITIVE: self.positive,
                TensorKey.NEGATIVE: self.negative,
                TensorKey.BASE_LATENT: self.base_latent,
                TensorKey.AUX_LATENT: self.aux_latent,
            }
            raw = mapping.get(key)
            return TensorValue(raw) if raw is not None else None

        self.tensor_pool.get.side_effect = _get_side_effect
        self.tensor_pool.peek.side_effect = _get_side_effect

        # mock model_pool
        self.model_pool = MagicMock()
        self.mock_diffusion_model = MagicMock()
        self.model_pool.get_model.return_value = self.mock_diffusion_model

        # mock device_ctx
        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

        # mock context
        self.sample_config = SampleConfig(steps=20, cfg_scale=5.0, solver=SolverType.EULER)
        self.runtime_config = RuntimeConfig(size=(512, 512), frame_num=16, seed=42)
        self.context = NodeContext(
            sample_config=self.sample_config,
            runtime_config=self.runtime_config,
            metadata={"noise_shape": [4, 16, 32, 32]},
        )

    @patch("kdit.nodes.infers.generator_node.GeneratorRunner")
    @patch("kdit.nodes.infers.generator_node.get_generator_def")
    def test_run_constructs_context_and_calls_generator(self, mock_get_def, mock_runner_cls):
        """run() 应构造 GeneratorInferContext 并调用 runner.run(ctx)。"""
        mock_runner = MagicMock()
        mock_runner.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_runner_cls.return_value = mock_runner

        self.node.run(
            model_key=ModelKey.Wan2_2_T2V_14B,
            context=self.context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )

        # 验证 get_generator_def 被调用
        mock_get_def.assert_called_once_with(ModelKey.Wan2_2_T2V_14B)
        # 验证 GeneratorRunner 被构造
        mock_runner_cls.assert_called_once_with(mock_get_def.return_value)
        # 验证 runner.run 被调用且参数是 GeneratorInferContext
        mock_runner.run.assert_called_once()
        ctx_arg = mock_runner.run.call_args[0][0]
        self.assertIsInstance(ctx_arg, GeneratorInferContext)
        self.assertIs(ctx_arg.diffusion_model, self.mock_diffusion_model)
        self.assertIsNotNone(ctx_arg.base_latent)
        self.assertIs(ctx_arg.sample_config, self.sample_config)
        self.assertIs(ctx_arg.runtime_config, self.runtime_config)

        # 验证 latents 写入 tensor_pool
        self.tensor_pool.put.assert_called_once()
        put_args = self.tensor_pool.put.call_args
        self.assertEqual(put_args[0][0], TensorKey.LATENTS)

    @patch("kdit.nodes.infers.generator_node.GeneratorRunner")
    @patch("kdit.nodes.infers.generator_node.get_generator_def")
    def test_base_latent_constructed_from_tensor_pool(self, mock_get_def, mock_runner_cls):
        """base_latent 应从 tensor_pool 中的 BASE_LATENT 构造为 BaseLatent 对象。"""
        from kdit.generators.generator_context import BaseLatent

        mock_runner = MagicMock()
        mock_runner.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_runner_cls.return_value = mock_runner

        context_no_shape = NodeContext(
            sample_config=self.sample_config,
            runtime_config=self.runtime_config,
            metadata={},
        )

        self.node.run(
            model_key=ModelKey.Wan2_2_T2V_14B,
            context=context_no_shape,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )

        ctx_arg = mock_runner.run.call_args[0][0]
        self.assertIsInstance(ctx_arg.base_latent, BaseLatent)
        # base_latent.latent 应为 tensor_pool 中 BASE_LATENT 的第一个元素
        self.assertTrue(torch.equal(ctx_arg.base_latent.latent, self.base_latent[0]))

    def test_dispatch_policy(self):
        self.assertEqual(GeneratorNode.dispatch_policy, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_tensor_keys(self):
        self.assertIn(TensorKey.POSITIVE, GeneratorNode.input_tensor_keys)
        self.assertIn(TensorKey.NEGATIVE, GeneratorNode.input_tensor_keys)
        self.assertIn(TensorKey.BASE_LATENT, GeneratorNode.input_tensor_keys)
        self.assertIn(TensorKey.AUX_LATENT, GeneratorNode.input_tensor_keys)
        self.assertEqual(GeneratorNode.output_tensor_keys, [TensorKey.LATENTS])


if __name__ == "__main__":
    unittest.main()
