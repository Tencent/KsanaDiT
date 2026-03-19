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

"""Tests for kdit.nodes.infers.vae_encoder_node — VAEEncodeSpatialNode / VAEEncodeImagesNode。

使用 mock 替代 model_pool / tensor_pool，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.models.model_key import ModelKey
from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import NodeDispatchPolicy
from kdit.nodes.infers.vae_encoder_node import VAEEncodeImagesNode, VAEEncodeSpatialNode
from kdit.tensor import TensorKey


class TestVAEEncodeSpatialNode(unittest.TestCase):
    """VAEEncodeSpatialNode 编码首尾帧为视频 latent。"""

    def setUp(self):
        self.node = VAEEncodeSpatialNode()
        self.start_img = torch.randn(1, 3, 480, 832)
        self.end_img = torch.randn(1, 3, 480, 832)

        self.tensor_pool = MagicMock()

        def _get_data(key):
            if key == TensorKey.START_IMG:
                return self.start_img
            if key == TensorKey.END_IMG:
                return self.end_img
            return None

        self.tensor_pool.get.side_effect = _get_data
        self.tensor_pool.peek.side_effect = _get_data

        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.model_key = ModelKey.VAE_WAN2_2
        self.mock_vae.forward_encode.return_value = (torch.randn(1, 16, 16, 32, 32), None)
        self.model_pool.get_model.return_value = self.mock_vae

        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

    def test_run_calls_forward_encode(self):
        context = NodeContext(metadata={"target_f": 16, "target_h": 480, "target_w": 832})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.mock_vae.forward_encode.assert_called_once()

    def test_run_writes_base_latent_as_list(self):
        context = NodeContext(metadata={"target_f": 16, "target_h": 480, "target_w": 832})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.tensor_pool.put.assert_called_once()
        put_key, put_val = self.tensor_pool.put.call_args[0]
        self.assertEqual(put_key, TensorKey.BASE_LATENT)
        self.assertIsInstance(put_val, list)

    def test_none_encode_result_not_written(self):
        self.mock_vae.forward_encode.return_value = (None, None)
        context = NodeContext(metadata={"target_f": 16, "target_h": 480, "target_w": 832})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.tensor_pool.put.assert_not_called()

    def test_dispatch_policy(self):
        self.assertEqual(VAEEncodeSpatialNode.dispatch_policy, NodeDispatchPolicy.R0_R0_BCAST)

    def test_tensor_keys(self):
        self.assertIn(TensorKey.START_IMG, VAEEncodeSpatialNode.input_tensor_keys)
        self.assertIn(TensorKey.END_IMG, VAEEncodeSpatialNode.input_tensor_keys)
        self.assertEqual(VAEEncodeSpatialNode.output_tensor_keys, [TensorKey.BASE_LATENT])


class TestVAEEncodeImagesNode(unittest.TestCase):
    """VAEEncodeImagesNode 编码参考图为 latent。"""

    def setUp(self):
        self.node = VAEEncodeImagesNode()
        self.image = torch.randn(1, 3, 480, 832)

        self.tensor_pool = MagicMock()
        self.tensor_pool.get.side_effect = lambda key: self.image if key == TensorKey.IMAGE else None
        self.tensor_pool.peek.side_effect = lambda key: self.image if key == TensorKey.IMAGE else None

        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.model_key = ModelKey.VAE_WAN2_2
        self.mock_vae.forward_encode_image.return_value = torch.randn(1, 16, 1, 32, 32)
        self.model_pool.get_model.return_value = self.mock_vae

        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

    def test_run_calls_forward_encode_image(self):
        context = NodeContext(metadata={"batch_size": 1})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.mock_vae.forward_encode_image.assert_called_once()

    def test_run_writes_image_embeds_as_list(self):
        context = NodeContext(metadata={"batch_size": 1})
        self.node.run(
            model_key=ModelKey.VAE_WAN2_2,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )
        self.tensor_pool.put.assert_called_once()
        put_key, put_val = self.tensor_pool.put.call_args[0]
        self.assertEqual(put_key, TensorKey.IMAGE_EMBEDS)
        self.assertIsInstance(put_val, list)

    def test_dispatch_policy(self):
        self.assertEqual(VAEEncodeImagesNode.dispatch_policy, NodeDispatchPolicy.R0_R0_BCAST)

    def test_tensor_keys(self):
        self.assertEqual(VAEEncodeImagesNode.input_tensor_keys, [TensorKey.IMAGE])
        self.assertEqual(VAEEncodeImagesNode.output_tensor_keys, [TensorKey.IMAGE_EMBEDS])


if __name__ == "__main__":
    unittest.main()
