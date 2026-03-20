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

"""QwenTextHandler 单元测试。

测试覆盖:
- preprocess (tuple / bare tensor / invalid)
- validate (mask 维度校验)
- cast_to (embeds + mask 设备/dtype 转换)
- expand_to_batch (tuple 扩展)
- _pad_text_pair (不等长 padding)
"""

import unittest

import torch

from kdit.generators.handlers_impl.qwen_text import QwenTextHandler


class TestQwenPreprocess(unittest.TestCase):
    """测试 preprocess — tuple / bare tensor / invalid。"""

    def test_tuple_passthrough(self):
        handler = QwenTextHandler()
        embeds = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.long)
        result = handler.preprocess((embeds, mask))
        self.assertTrue(torch.equal(result[0], embeds))
        self.assertTrue(torch.equal(result[1], mask))

    def test_bare_tensor_generates_mask(self):
        handler = QwenTextHandler()
        embeds = torch.randn(2, 10, 768)
        result = handler.preprocess(embeds)
        self.assertEqual(result[0].shape, embeds.shape)
        self.assertEqual(result[1].shape, (2, 10))
        self.assertTrue((result[1] == 1).all())

    def test_invalid_format_raises(self):
        handler = QwenTextHandler()
        with self.assertRaises(ValueError):
            handler.preprocess(42)


class TestQwenValidate(unittest.TestCase):
    """测试 validate — mask 维度校验。"""

    def test_valid_prompts(self):
        handler = QwenTextHandler()
        pos = torch.randn(2, 10, 768)
        neg = torch.randn(2, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        result_pos, _ = handler.validate((pos, pos_mask), (neg, neg_mask))
        self.assertEqual(result_pos[0].shape[0], 2)

    def test_mismatched_batch_raises(self):
        handler = QwenTextHandler()
        pos = torch.randn(2, 10, 768)
        neg = torch.randn(3, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(3, 10, dtype=torch.long)
        with self.assertRaises(ValueError):
            handler.validate((pos, pos_mask), (neg, neg_mask))


class TestQwenCastTo(unittest.TestCase):
    """测试 cast_to — embeds + mask 转换。"""

    def test_cast_dtype_and_device(self):
        handler = QwenTextHandler()
        pos = torch.randn(2, 10, 768, dtype=torch.float32)
        neg = torch.randn(2, 10, 768, dtype=torch.float32)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        result_pos, result_neg = handler.cast_to(
            (pos, pos_mask),
            (neg, neg_mask),
            dtype=torch.float16,
            device=torch.device("cpu"),
        )
        self.assertEqual(result_pos[0].dtype, torch.float16)
        self.assertEqual(result_neg[0].dtype, torch.float16)
        self.assertEqual(result_pos[1].device, torch.device("cpu"))


class TestQwenExpandToBatch(unittest.TestCase):
    """测试 expand_to_batch — tuple 扩展。"""

    def test_expand_tuple(self):
        handler = QwenTextHandler()
        pos = torch.randn(2, 10, 768)
        neg = torch.randn(2, 10, 768)
        pos_mask = torch.ones(2, 10, dtype=torch.long)
        neg_mask = torch.ones(2, 10, dtype=torch.long)
        result_pos, result_neg = handler.expand_to_batch(
            (pos, pos_mask),
            (neg, neg_mask),
            batch_size_per_prompts=[2, 3],
        )
        self.assertEqual(result_pos[0].shape[0], 5)
        self.assertEqual(result_neg[0].shape[0], 5)
        self.assertEqual(result_pos[1].shape[0], 5)


class TestQwenPadTextPair(unittest.TestCase):
    """测试 _pad_text_pair — 不等长 padding。"""

    def test_equal_length_no_pad(self):
        handler = QwenTextHandler()
        a = torch.randn(2, 10, 768)
        ma = torch.ones(2, 10)
        b = torch.randn(2, 10, 768)
        mb = torch.ones(2, 10)
        ra, _, rb, _ = handler._pad_text_pair(a, ma, b, mb)
        self.assertEqual(ra.shape[1], 10)
        self.assertEqual(rb.shape[1], 10)

    def test_unequal_length_pads(self):
        handler = QwenTextHandler()
        a = torch.randn(2, 8, 768)
        ma = torch.ones(2, 8)
        b = torch.randn(2, 12, 768)
        mb = torch.ones(2, 12)
        ra, rma, rb, _ = handler._pad_text_pair(a, ma, b, mb)
        self.assertEqual(ra.shape[1], 12)
        self.assertEqual(rb.shape[1], 12)
        self.assertEqual(rma.shape[1], 12)


if __name__ == "__main__":
    unittest.main()
