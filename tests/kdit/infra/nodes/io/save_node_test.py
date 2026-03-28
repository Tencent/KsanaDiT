# Copyright 2026 Tencent
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

"""Tests for kdit.nodes.io.save_node — SaveVideoNode / SaveImageNode。

使用 mock PinHub 替代 tensor_pool / model_pool，不需要 GPU。
重点回归测试：传给 save_video 的 tensor 必须是 5D [B, C, T, H, W]，
而非 6D（历史 bug: video[None] 多加了一维）。
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import NodeDispatchPolicy
from kdit.nodes.io.save_node import SaveImageNode, SaveVideoNode
from kdit.tensor import TensorKey


def _make_pins(tensor_map: dict[TensorKey, torch.Tensor | None] | None = None):
    """构造 mock PinHub — 只需 get_tensor() 方法。"""
    pins = MagicMock()
    if tensor_map is None:
        tensor_map = {}
    pins.get_tensor.side_effect = lambda key: tensor_map.get(key)
    return pins


class TestSaveVideoNode(unittest.TestCase):
    """SaveVideoNode 从 PinHub 读取 VIDEO tensor 并调用 save_video。"""

    def setUp(self):
        self.node = SaveVideoNode()
        # 典型 VAE decode 输出: [B=1, C=3, T=49, H=512, W=512]
        self.video_5d = torch.randn(1, 3, 4, 8, 8)
        self.pins = _make_pins({TensorKey.VIDEO: self.video_5d})

    @patch("kdit.nodes.io.save_node.save_video")
    def test_passes_5d_tensor_to_save_video(self, mock_save_video):
        """回归测试：传给 save_video 的 tensor 必须是 5D，不能是 6D。"""
        context = NodeContext(metadata={"save_path": "/tmp/test_video.mp4", "fps": 24})
        self.node.run(self.pins, context=context)
        mock_save_video.assert_called_once()
        call_kwargs = mock_save_video.call_args[1]
        tensor_arg = call_kwargs["tensor"]
        self.assertEqual(tensor_arg.ndim, 5, f"save_video tensor must be 5D, got {tensor_arg.ndim}D")
        self.assertTrue(torch.equal(tensor_arg, self.video_5d))

    @patch("kdit.nodes.io.save_node.save_video")
    def test_passes_correct_save_params(self, mock_save_video):
        """验证 save_path / fps / normalize / value_range 正确传递。"""
        context = NodeContext(metadata={"save_path": "/tmp/out.mp4", "fps": 16})
        self.node.run(self.pins, context=context)
        call_kwargs = mock_save_video.call_args[1]
        self.assertEqual(call_kwargs["save_file"], "/tmp/out.mp4")
        self.assertEqual(call_kwargs["fps"], 16)
        self.assertEqual(call_kwargs["nrow"], 1)
        self.assertTrue(call_kwargs["normalize"])
        self.assertEqual(call_kwargs["value_range"], (-1, 1))

    @patch("kdit.nodes.io.save_node.save_video")
    def test_default_fps_is_30(self, mock_save_video):
        """metadata 中无 fps 时默认 30。"""
        context = NodeContext(metadata={"save_path": "/tmp/out.mp4"})
        self.node.run(self.pins, context=context)
        call_kwargs = mock_save_video.call_args[1]
        self.assertEqual(call_kwargs["fps"], 30)

    @patch("kdit.nodes.io.save_node.save_video")
    def test_skips_when_no_video_tensor(self, mock_save_video):
        """PinHub 中无 VIDEO tensor 时跳过保存。"""
        empty_pins = _make_pins()
        context = NodeContext(metadata={"save_path": "/tmp/out.mp4"})
        self.node.run(empty_pins, context=context)
        mock_save_video.assert_not_called()

    @patch("kdit.nodes.io.save_node.save_video")
    def test_skips_when_no_save_path(self, mock_save_video):
        """metadata 中无 save_path 时跳过保存。"""
        context = NodeContext(metadata={})
        self.node.run(self.pins, context=context)
        mock_save_video.assert_not_called()

    def test_dispatch_policy(self):
        self.assertEqual(SaveVideoNode.dispatch_policy, NodeDispatchPolicy.ALL_R0_R0)

    def test_tensor_pins(self):
        self.assertEqual(SaveVideoNode.input_defs, [TensorKey.VIDEO])
        self.assertEqual(SaveVideoNode.output_defs, [])


class TestSaveImageNode(unittest.TestCase):
    """SaveImageNode 从 PinHub 读取 VIDEO tensor 并调用 save_image。"""

    def setUp(self):
        self.node = SaveImageNode()
        self.image = torch.randn(1, 3, 256, 256)
        self.pins = _make_pins({TensorKey.VIDEO: self.image})

    @patch("kdit.nodes.io.save_node.save_image")
    def test_passes_tensor_to_save_image(self, mock_save_image):
        context = NodeContext(metadata={"save_path": "/tmp/test.png"})
        self.node.run(self.pins, context=context)
        mock_save_image.assert_called_once()
        call_kwargs = mock_save_image.call_args[1]
        self.assertTrue(torch.equal(call_kwargs["tensor"], self.image))
        self.assertEqual(call_kwargs["path"], "/tmp/test.png")

    @patch("kdit.nodes.io.save_node.save_image")
    def test_skips_when_no_image_tensor(self, mock_save_image):
        empty_pins = _make_pins()
        context = NodeContext(metadata={"save_path": "/tmp/test.png"})
        self.node.run(empty_pins, context=context)
        mock_save_image.assert_not_called()

    @patch("kdit.nodes.io.save_node.save_image")
    def test_skips_when_no_save_path(self, mock_save_image):
        context = NodeContext(metadata={})
        self.node.run(self.pins, context=context)
        mock_save_image.assert_not_called()

    def test_dispatch_policy(self):
        self.assertEqual(SaveImageNode.dispatch_policy, NodeDispatchPolicy.ALL_R0_R0)

    def test_tensor_pins(self):
        self.assertEqual(SaveImageNode.input_defs, [TensorKey.VIDEO])
        self.assertEqual(SaveImageNode.output_defs, [])


if __name__ == "__main__":
    unittest.main()
