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

"""Tests for kdit.nodes.infers.vae_decoder_node — VAEDecodeNode.run()。

使用 mock 替代 model_pool / tensor_pool，通过 PinHub 访问数据，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.models.model_key import ModelKey
from kdit.models.model_pool_key import ModelPoolKey
from kdit.nodes.core.device_info import DeviceInfo
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType, NodeDispatchPolicy
from kdit.nodes.core.pin_hub import PinHub
from kdit.nodes.infers.vae_decoder_node import VAEDecodeNode
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey


def _make_pins(*, node_id=0, model_key=None, tensor_pool=None, model_pool=None, tensor_mapping=None):
    """构建一个 PinHub 实例。"""
    input_pins = dict(tensor_mapping or {})
    if model_key is not None:
        input_pins[model_key] = ModelPoolKey(99, model_key)
    node_def = NodeDef(node_id=node_id, node_type=InferNodeType.VAE_DECODE, model_key=model_key)
    return PinHub(
        node_def=node_def,
        input_pins=input_pins,
        tensor_pool=tensor_pool or MagicMock(),
        model_pool=model_pool or MagicMock(),
    )


class TestVAEDecodeNode(unittest.TestCase):
    """VAEDecodeNode 从 PinHub 读取 latents，调用 vae_model.forward_decode()。"""

    def setUp(self):
        self.node = VAEDecodeNode()
        self.node._factory_model_key = ModelKey.VAE_WAN2_2
        self.latents = torch.randn(1, 4, 16, 32, 32)

        # 真实 TensorPool + 预填充数据
        self.tensor_pool = TensorPool()
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.LATENTS), self.latents)

        # mock model_pool
        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.forward_decode.return_value = torch.randn(1, 3, 16, 256, 256)
        self.model_pool.get_model.return_value = self.mock_vae

        # device_info
        self.device_info = DeviceInfo(
            compute_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

        self.tensor_mapping = {TensorKey.LATENTS: TensorPoolKey(10, TensorKey.LATENTS)}

    def _make_pins(self, node_id=0):
        return _make_pins(
            node_id=node_id,
            model_key=ModelKey.VAE_WAN2_2,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            tensor_mapping=self.tensor_mapping,
        )

    def test_run_calls_forward_decode(self):
        context = NodeContext(device=self.device_info, metadata={})
        pins = self._make_pins()
        self.node.run(pins, context=context)
        self.mock_vae.forward_decode.assert_called_once()
        call_kwargs = self.mock_vae.forward_decode.call_args[1]
        self.assertTrue(torch.equal(call_kwargs["latents"], self.latents))
        self.assertEqual(call_kwargs["local_rank"], 0)

    def test_run_writes_video_to_tensor_pool(self):
        context = NodeContext(device=self.device_info, metadata={})
        pins = self._make_pins(node_id=5)
        self.node.run(pins, context=context)
        # 验证 VIDEO 写入 tensor_pool（通过 PinHub → TensorPoolKey(5, VIDEO)）
        tv = self.tensor_pool.get(TensorPoolKey(5, TensorKey.VIDEO))
        self.assertIsNotNone(tv)

    def test_offload_model_when_requested(self):
        context = NodeContext(
            device=self.device_info,
            metadata={"offload_model": True},
        )
        pins = self._make_pins()
        self.node.run(pins, context=context)
        self.mock_vae.to.assert_called_once_with(torch.device("cpu"))

    def test_no_offload_by_default(self):
        context = NodeContext(device=self.device_info, metadata={})
        pins = self._make_pins()
        self.node.run(pins, context=context)
        self.mock_vae.to.assert_not_called()

    def test_dispatch_policy(self):
        self.assertEqual(VAEDecodeNode.dispatch_policy, NodeDispatchPolicy.ALL_R0_R0)

    def test_tensor_pins(self):
        self.assertEqual(VAEDecodeNode.input_defs, [TensorKey.LATENTS])
        self.assertEqual(VAEDecodeNode.output_defs, [TensorKey.VIDEO])


if __name__ == "__main__":
    unittest.main()
