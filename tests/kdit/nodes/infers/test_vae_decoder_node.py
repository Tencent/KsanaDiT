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

使用 mock 替代 model_pool / tensor_pool，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.models.model_key import ModelKey
from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import NodeDispatchPolicy
from kdit.nodes.infers.vae_decoder_node import VAEDecodeNode
from kdit.tensor import TensorKey


class TestVAEDecodeNode(unittest.TestCase):
    """VAEDecodeNode 从 tensor_pool 读取 latents，调用 vae_model.forward_decode()。"""

    def setUp(self):
        self.node = VAEDecodeNode()
        self.latents = torch.randn(1, 4, 16, 32, 32)

        # mock tensor_pool
        self.tensor_pool = MagicMock()
        self.tensor_pool.get.side_effect = lambda key: self.latents if key == TensorKey.LATENTS else None
        self.tensor_pool.peek.side_effect = lambda key: self.latents if key == TensorKey.LATENTS else None

        # mock model_pool
        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.forward_decode.return_value = torch.randn(1, 3, 16, 256, 256)
        self.model_pool.get_model.return_value = self.mock_vae

        # device_ctx
        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

    def test_run_calls_forward_decode(self):
        context = NodeContext(metadata={})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.mock_vae.forward_decode.assert_called_once()
        call_kwargs = self.mock_vae.forward_decode.call_args[1]
        self.assertTrue(torch.equal(call_kwargs["latents"], self.latents))
        self.assertEqual(call_kwargs["local_rank"], 0)

    def test_run_writes_video_to_tensor_pool(self):
        context = NodeContext(metadata={})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.tensor_pool.put.assert_called_once()
        put_key = self.tensor_pool.put.call_args[0][0]
        self.assertEqual(put_key, TensorKey.VIDEO)

    def test_offload_model_when_requested(self):
        context = NodeContext(metadata={"offload_model": True})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.mock_vae.to.assert_called_once_with(torch.device("cpu"))

    def test_no_offload_by_default(self):
        context = NodeContext(metadata={})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.mock_vae.to.assert_not_called()

    def test_dispatch_policy(self):
        self.assertEqual(VAEDecodeNode.dispatch_policy, NodeDispatchPolicy.ALL_R0_R0)

    def test_tensor_keys(self):
        self.assertEqual(VAEDecodeNode.input_tensor_keys, [TensorKey.LATENTS])
        self.assertEqual(VAEDecodeNode.output_tensor_keys, [TensorKey.VIDEO])


if __name__ == "__main__":
    unittest.main()
