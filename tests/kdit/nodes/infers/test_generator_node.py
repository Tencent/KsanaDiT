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

使用 mock 替代 model_pool / tensor_pool / GeneratorFactory，不需要 GPU。
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

    @patch("kdit.nodes.infers.generator_node.GeneratorFactory")
    def test_run_constructs_context_and_calls_generator(self, mock_factory):
        """run() 应构造 GeneratorInferContext 并调用 generator.run(ctx)。"""
        mock_generator = MagicMock()
        mock_generator.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_factory.create.return_value = mock_generator

        self.node.run(
            model_key=ModelKey.Wan2_2_T2V_14B,
            context=self.context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )

        # 验证 generator.run 被调用且参数是 GeneratorInferContext
        mock_generator.run.assert_called_once()
        ctx_arg = mock_generator.run.call_args[0][0]
        self.assertIsInstance(ctx_arg, GeneratorInferContext)
        self.assertIs(ctx_arg.diffusion_model, self.mock_diffusion_model)
        self.assertEqual(ctx_arg.noise_shape, [4, 16, 32, 32])
        self.assertIs(ctx_arg.sample_config, self.sample_config)
        self.assertIs(ctx_arg.runtime_config, self.runtime_config)

        # 验证 latents 写入 tensor_pool
        self.tensor_pool.put.assert_called_once()
        put_args = self.tensor_pool.put.call_args
        self.assertEqual(put_args[0][0], TensorKey.LATENTS)

    @patch("kdit.nodes.infers.generator_node.GeneratorFactory")
    def test_noise_shape_from_base_latent_when_metadata_missing(self, mock_factory):
        """当 metadata 中没有 noise_shape 时，应从 base_latent 推导。"""
        mock_generator = MagicMock()
        mock_generator.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_factory.create.return_value = mock_generator

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

        ctx_arg = mock_generator.run.call_args[0][0]
        # base_latent[0].shape = (1, 4, 16, 32, 32), shape[1:] = (4, 16, 32, 32)
        self.assertEqual(ctx_arg.noise_shape, [4, 16, 32, 32])

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
