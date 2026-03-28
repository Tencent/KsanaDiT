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

"""Tests for kdit.nodes.infer.generator_node — GeneratorNode.run()。

使用 mock 替代 PinHub 内部的 pool / GeneratorRunner，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.config import RuntimeConfig, SampleConfig, SolverType
from kdit.generators.generator_context import GeneratorInferContext
from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.device_info import DeviceInfo
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType, NodeDispatchPolicy
from kdit.nodes.core.pin_hub import PinHub
from kdit.nodes.infer.generator_node import GeneratorNode
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey


class TestGeneratorNode(unittest.TestCase):
    """GeneratorNode 从 PinHub 读取数据，构造 GeneratorInferContext 并调用 generator.run()。"""

    def setUp(self):
        self.node = GeneratorNode()
        self.node._factory_model_key = ModelKey.Wan2_2_T2V_14B

        # 真实 TensorPool + 预填充数据
        self.tensor_pool = TensorPool()
        self.positive = torch.randn(1, 77, 768)
        self.negative = torch.randn(1, 77, 768)
        self.base_latent = [torch.randn(1, 4, 16, 32, 32)]
        self.aux_latent = torch.randn(1, 4, 16, 32, 32)

        # 用上游 node_id=10 写入 tensor
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.POSITIVE), self.positive)
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.NEGATIVE), self.negative)
        self.tensor_pool.put(TensorPoolKey(20, TensorKey.BASE_LATENT), self.base_latent)
        self.tensor_pool.put(TensorPoolKey(20, TensorKey.AUX_LATENT), self.aux_latent)

        # mock model_pool
        self.model_pool = MagicMock()
        self.mock_diffusion_model = MagicMock()
        self.model_pool.get_model.return_value = self.mock_diffusion_model

        # device_info
        self.device_info = DeviceInfo(
            compute_device=torch.device("cpu"),
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
            device=self.device_info,
            metadata={"noise_shape": [4, 16, 32, 32]},
        )

        # input_pins: 映射上游 tensor/model（扁平 Pins 格式）
        self.input_pins = {
            ModelKey.Wan2_2_T2V_14B: ModelPoolKey(99, ModelKey.Wan2_2_T2V_14B),
            TensorKey.POSITIVE: TensorPoolKey(10, TensorKey.POSITIVE),
            TensorKey.NEGATIVE: TensorPoolKey(10, TensorKey.NEGATIVE),
            TensorKey.BASE_LATENT: TensorPoolKey(20, TensorKey.BASE_LATENT),
            TensorKey.AUX_LATENT: TensorPoolKey(20, TensorKey.AUX_LATENT),
        }

    def _make_pins(self, input_pins=None):
        node_def = NodeDef(node_type=InferNodeType.GENERATE, model_key=ModelKey.Wan2_2_T2V_14B)
        self._node_def = node_def
        return PinHub(
            node_def=node_def,
            input_pins=input_pins or self.input_pins,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
        )

    @patch("kdit.nodes.infer.generator_node.GeneratorRunner")
    @patch("kdit.nodes.infer.generator_node.get_generator_def")
    def test_run_constructs_context_and_calls_generator(self, mock_get_def, mock_runner_cls):
        """run() 应构造 GeneratorInferContext 并调用 runner.run(ctx)。"""
        mock_runner = MagicMock()
        mock_runner.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_runner_cls.return_value = mock_runner

        pins = self._make_pins()
        self.node.run(pins, context=self.context)

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

        # 验证 latents 写入 tensor_pool（通过 PinHub → TensorPoolKey(node_id, LATENTS)）
        tv = self.tensor_pool.get(TensorPoolKey(self._node_def.node_id, TensorKey.LATENTS))
        self.assertIsNotNone(tv)

    @patch("kdit.nodes.infer.generator_node.GeneratorRunner")
    @patch("kdit.nodes.infer.generator_node.get_generator_def")
    def test_base_latent_constructed_from_tensor_pool(self, mock_get_def, mock_runner_cls):
        """base_latent 应从 PinHub 中的 BASE_LATENT 构造为 BaseLatent 对象。"""
        from kdit.generators.generator_context import BaseLatent

        mock_runner = MagicMock()
        mock_runner.run.return_value = torch.randn(1, 4, 16, 32, 32)
        mock_runner_cls.return_value = mock_runner

        context_no_shape = NodeContext(
            sample_config=self.sample_config,
            runtime_config=self.runtime_config,
            device=self.device_info,
            metadata={},
        )

        pins = self._make_pins()
        self.node.run(pins, context=context_no_shape)

        ctx_arg = mock_runner.run.call_args[0][0]
        self.assertIsInstance(ctx_arg.base_latent, BaseLatent)
        # base_latent.latent 应为 tensor_pool 中 BASE_LATENT 的第一个元素
        self.assertTrue(torch.equal(ctx_arg.base_latent.latent, self.base_latent[0]))

    def test_dispatch_policy(self):
        self.assertEqual(GeneratorNode.dispatch_policy, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_tensor_pins(self):
        self.assertIn(TensorKey.POSITIVE, GeneratorNode.input_defs)
        self.assertIn(TensorKey.NEGATIVE, GeneratorNode.input_defs)
        self.assertIn(TensorKey.BASE_LATENT, GeneratorNode.input_defs)
        self.assertIn(TensorKey.AUX_LATENT, GeneratorNode.input_defs)
        self.assertEqual(GeneratorNode.output_defs, [TensorKey.LATENTS])


if __name__ == "__main__":
    unittest.main()
