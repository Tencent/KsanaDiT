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

"""Tests for kdit.generators.steps.tensor_ops — split_tensors / cast_to."""

import unittest

import torch

from kdit.generators.steps.tensor_ops import cast_to, split_tensors


class TestSplitTensors(unittest.TestCase):
    """split_tensors 按 batch 维度切片。"""

    def test_split_tensor(self):
        t = torch.arange(20).reshape(4, 5)
        result = split_tensors(t, 1, 3)
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(torch.equal(result, t[1:3]))

    def test_split_tuple(self):
        a = torch.zeros(4, 3)
        b = torch.ones(4, 3)
        result = split_tensors((a, b), 0, 2)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (2, 3))
        self.assertEqual(result[1].shape, (2, 3))

    def test_split_list(self):
        a = torch.zeros(4, 3)
        b = torch.ones(4, 3)
        result = split_tensors([a, b], 1, 4)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0].shape, (3, 3))

    def test_split_none_returns_none(self):
        self.assertIsNone(split_tensors(None, 0, 2))

    def test_split_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            split_tensors("invalid", 0, 1)


class TestCastTo(unittest.TestCase):
    """cast_to 转换 dtype 和 device。"""

    def test_cast_dtype(self):
        t = torch.ones(2, 3, dtype=torch.float32)
        result = cast_to(t, dtype=torch.float16, device=torch.device("cpu"))
        self.assertEqual(result.dtype, torch.float16)

    def test_cast_same_dtype_no_copy(self):
        t = torch.ones(2, 3, dtype=torch.float32)
        result = cast_to(t, dtype=torch.float32, device=torch.device("cpu"))
        # 当 dtype 和 device 都相同时，应该返回原始 tensor（无拷贝）
        self.assertIs(result, t)

    def test_cast_device_cpu_to_cpu(self):
        t = torch.ones(2, 3, dtype=torch.float32, device="cpu")
        result = cast_to(t, dtype=torch.float32, device=torch.device("cpu"))
        self.assertEqual(result.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
