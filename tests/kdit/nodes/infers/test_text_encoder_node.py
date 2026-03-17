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

"""Tests for kdit.nodes.infers.text_encoder_node — T5TextEncodeNode / QwenTextEncodeNode。

使用 mock 替代 model_pool / tensor_pool，不需要 GPU。
"""

import unittest
from unittest.mock import MagicMock

import torch

from kdit.models.model_key import ModelKey
from kdit.nodes.core.device_context import NodeDeviceContext
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import NodeDispatchPolicy
from kdit.nodes.infers.text_encoder_node import QwenTextEncodeNode, T5TextEncodeNode
from kdit.tensor import TensorKey


class TestT5TextEncodeNode(unittest.TestCase):
    """T5TextEncodeNode 调用 model.forward() 并写入 positive/negative。"""

    def setUp(self):
        self.node = T5TextEncodeNode()

        self.tensor_pool = MagicMock()
        self.model_pool = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.device = torch.device("cpu")
        self.mock_model.default_settings = MagicMock(neg_prompt="")
        self.model_pool.get_model.return_value = self.mock_model

        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

    def test_run_calls_model_forward_and_writes_tensors(self):
        # T5 forward 返回 list[Tensor]（pos + neg 合并后的列表）
        emb = torch.randn(77, 768)
        self.mock_model.forward.return_value = [emb, emb]

        context = NodeContext(
            prompt=["a cat"],
            negative_prompt=[""],
            metadata={},
        )

        self.node.run(
            model_key=ModelKey.T5TextEncoder,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )

        # 验证 model.forward 被调用（pos + neg 合并）
        self.mock_model.forward.assert_called_once_with(["a cat", ""])

        # 验证 positive/negative 写入 tensor_pool
        self.assertEqual(self.tensor_pool.put.call_count, 2)
        calls = self.tensor_pool.put.call_args_list
        self.assertEqual(calls[0][0][0], TensorKey.POSITIVE)
        self.assertEqual(calls[1][0][0], TensorKey.NEGATIVE)

    def test_dispatch_policy(self):
        self.assertEqual(T5TextEncodeNode.dispatch_policy, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_tensor_keys(self):
        self.assertEqual(T5TextEncodeNode.input_tensor_keys, [])
        self.assertIn(TensorKey.POSITIVE, T5TextEncodeNode.output_tensor_keys)
        self.assertIn(TensorKey.NEGATIVE, T5TextEncodeNode.output_tensor_keys)


class TestQwenTextEncodeNode(unittest.TestCase):
    """QwenTextEncodeNode 分别 forward pos/neg 并写入 (embeds, mask) 元组。"""

    def setUp(self):
        self.node = QwenTextEncodeNode()

        self.tensor_pool = MagicMock()
        self.model_pool = MagicMock()
        self.mock_model = MagicMock()
        self.mock_model.device = torch.device("cpu")
        self.mock_model.default_settings = MagicMock(neg_prompt="")
        self.model_pool.get_model.return_value = self.mock_model

        self.device_ctx = NodeDeviceContext(
            device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            rank_id=0,
            world_size=1,
        )

    def test_run_calls_model_forward_twice_and_writes_tuples(self):
        pos_embeds = torch.randn(1, 77, 768)
        pos_mask = torch.ones(1, 77)
        neg_embeds = torch.randn(1, 77, 768)
        neg_mask = torch.ones(1, 77)
        self.mock_model.forward.side_effect = [(pos_embeds, pos_mask), (neg_embeds, neg_mask)]

        context = NodeContext(
            prompt=["a cat"],
            negative_prompt=[""],
            metadata={},
        )

        self.node.run(
            model_key=ModelKey.Qwen2VLTextEncoder,
            context=context,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
            device_ctx=self.device_ctx,
        )

        # 验证 model.forward 被调用两次（pos 和 neg 分别）
        self.assertEqual(self.mock_model.forward.call_count, 2)

        # 验证 positive/negative 写入 tensor_pool（元组形式）
        self.assertEqual(self.tensor_pool.put.call_count, 2)
        calls = self.tensor_pool.put.call_args_list
        self.assertEqual(calls[0][0][0], TensorKey.POSITIVE)
        self.assertEqual(calls[1][0][0], TensorKey.NEGATIVE)
        # 验证写入的是 (embeds, mask) 元组
        self.assertIsInstance(calls[0][0][1], tuple)
        self.assertIsInstance(calls[1][0][1], tuple)

    def test_dispatch_policy(self):
        self.assertEqual(QwenTextEncodeNode.dispatch_policy, NodeDispatchPolicy.ALL_ALL_ALL)

    def test_tensor_keys(self):
        self.assertEqual(QwenTextEncodeNode.input_tensor_keys, [])
        self.assertIn(TensorKey.POSITIVE, QwenTextEncodeNode.output_tensor_keys)
        self.assertIn(TensorKey.NEGATIVE, QwenTextEncodeNode.output_tensor_keys)


if __name__ == "__main__":
    unittest.main()
