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
from kdit.nodes.infers.vae_encoder_node import VAEEncodeImagesNode, VAEEncodeSpatialNode
from kdit.pipelines.pipeline_def import NodeDef
from kdit.tensor import TensorKey
from kdit.tensor.tensor_pool import TensorPool
from kdit.tensor.tensor_pool_key import TensorPoolKey


class TestVAEEncodeSpatialNode(unittest.TestCase):
    """VAEEncodeSpatialNode 编码首尾帧为视频 latent。"""

    def setUp(self):
        self.node = VAEEncodeSpatialNode()
        self.node._factory_model_key = ModelKey.VAE_WAN2_2
        self.start_img = torch.randn(1, 3, 480, 832)
        self.end_img = torch.randn(1, 3, 480, 832)

        # 真实 TensorPool + 预填充数据
        self.tensor_pool = TensorPool()
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.START_IMG), self.start_img)
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.END_IMG), self.end_img)

        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.model_key = ModelKey.VAE_WAN2_2
        self.mock_vae.forward_encode.return_value = (torch.randn(1, 16, 16, 32, 32), None)
        self.model_pool.get_model.return_value = self.mock_vae

        self.device_info = DeviceInfo(
            compute_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

        self.tensor_mapping = {
            TensorKey.START_IMG: TensorPoolKey(10, TensorKey.START_IMG),
            TensorKey.END_IMG: TensorPoolKey(10, TensorKey.END_IMG),
        }

    def _make_pins(self, node_id=0):
        node_def = NodeDef(node_id=node_id, node_type=InferNodeType.VAE_ENCODE_SPATIAL, model_key=ModelKey.VAE_WAN2_2)
        return PinHub(
            node_def=node_def,
            input_pins={
                ModelKey.VAE_WAN2_2: ModelPoolKey(99, ModelKey.VAE_WAN2_2),
                **self.tensor_mapping,
            },
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
        )

    def test_run_calls_forward_encode(self):
        context = NodeContext(
            device=self.device_info,
            metadata={"target_f": 16, "target_h": 480, "target_w": 832},
        )
        pins = self._make_pins()
        self.node.run(pins, context=context)
        self.mock_vae.forward_encode.assert_called_once()

    def test_run_writes_base_latent_as_list(self):
        context = NodeContext(
            device=self.device_info,
            metadata={"target_f": 16, "target_h": 480, "target_w": 832},
        )
        pins = self._make_pins(node_id=5)
        self.node.run(pins, context=context)
        # 验证 BASE_LATENT 写入 tensor_pool（通过 PinHub → TensorPoolKey(5, BASE_LATENT)）
        tv = self.tensor_pool.get(TensorPoolKey(5, TensorKey.BASE_LATENT))
        self.assertIsNotNone(tv)
        self.assertIsInstance(tv.data, list)

    def test_none_encode_result_not_written(self):
        self.mock_vae.forward_encode.return_value = (None, None)
        context = NodeContext(
            device=self.device_info,
            metadata={"target_f": 16, "target_h": 480, "target_w": 832},
        )
        pins = self._make_pins(node_id=7)
        self.node.run(pins, context=context)
        # latent 为 None 时不写入
        tv = self.tensor_pool.get(TensorPoolKey(7, TensorKey.BASE_LATENT))
        self.assertIsNone(tv)

    def test_dispatch_policy(self):
        self.assertEqual(VAEEncodeSpatialNode.dispatch_policy, NodeDispatchPolicy.R0_R0_BCAST)

    def test_tensor_pins(self):
        self.assertIn(TensorKey.START_IMG, VAEEncodeSpatialNode.input_defs)
        self.assertIn(TensorKey.END_IMG, VAEEncodeSpatialNode.input_defs)
        self.assertEqual(VAEEncodeSpatialNode.output_defs, [TensorKey.BASE_LATENT])


class TestVAEEncodeImagesNode(unittest.TestCase):
    """VAEEncodeImagesNode 编码参考图为 latent。"""

    def setUp(self):
        self.node = VAEEncodeImagesNode()
        self.node._factory_model_key = ModelKey.VAE_WAN2_2
        self.image = torch.randn(1, 3, 480, 832)

        # 真实 TensorPool + 预填充数据
        self.tensor_pool = TensorPool()
        self.tensor_pool.put(TensorPoolKey(10, TensorKey.IMAGE), self.image)

        self.model_pool = MagicMock()
        self.mock_vae = MagicMock()
        self.mock_vae.model_key = ModelKey.VAE_WAN2_2
        self.mock_vae.forward_encode_image.return_value = torch.randn(1, 16, 1, 32, 32)
        self.model_pool.get_model.return_value = self.mock_vae

        self.device_info = DeviceInfo(
            compute_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

        self.tensor_mapping = {TensorKey.IMAGE: TensorPoolKey(10, TensorKey.IMAGE)}

    def _make_pins(self, node_id=0):
        node_def = NodeDef(node_id=node_id, node_type=InferNodeType.VAE_ENCODE_IMAGES, model_key=ModelKey.VAE_WAN2_2)
        return PinHub(
            node_def=node_def,
            input_pins={
                ModelKey.VAE_WAN2_2: ModelPoolKey(99, ModelKey.VAE_WAN2_2),
                **self.tensor_mapping,
            },
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
        )

    def test_run_calls_forward_encode_image(self):
        context = NodeContext(
            device=self.device_info,
            metadata={"batch_size": 1},
        )
        pins = self._make_pins()
        self.node.run(pins, context=context)
        self.mock_vae.forward_encode_image.assert_called_once()

    def test_run_writes_aux_latent(self):
        context = NodeContext(
            device=self.device_info,
            metadata={"batch_size": 1},
        )
        pins = self._make_pins(node_id=5)
        self.node.run(pins, context=context)
        # 验证 AUX_LATENT 写入 tensor_pool（通过 PinHub → TensorPoolKey(5, AUX_LATENT)）
        tv = self.tensor_pool.get(TensorPoolKey(5, TensorKey.AUX_LATENT))
        self.assertIsNotNone(tv)

    def test_dispatch_policy(self):
        self.assertEqual(VAEEncodeImagesNode.dispatch_policy, NodeDispatchPolicy.R0_R0_BCAST)

    def test_tensor_pins(self):
        self.assertEqual(VAEEncodeImagesNode.input_defs, [TensorKey.IMAGE])
        self.assertEqual(VAEEncodeImagesNode.output_defs, [TensorKey.AUX_LATENT])


if __name__ == "__main__":
    unittest.main()
