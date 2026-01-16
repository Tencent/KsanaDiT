import unittest
from unittest.mock import patch

import torch

from ksana.models.model_key import KsanaModelKey
from ksana.scheduler.scheduler import KsanaScheduler


class TestKsanaScheduler(unittest.TestCase):
    """测试KsanaScheduler类"""

    def setUp(self):
        """设置测试环境"""
        self.scheduler = KsanaScheduler()
        self.latent_shape = [1, 16, 8, 64, 64]  # [batch_size, z_dim, frames, height, width]
        self.device = torch.device("cuda:0")

    @patch("ksana.scheduler.scheduler.get_available_memory")
    def test_build_batch_strategy_sufficient_memory(self, mock_get_memory):
        """测试内存充足时的批处理策略"""
        # 模拟充足的内存
        mock_get_memory.return_value = 100 * 1024 * 1024 * 1024  # 100GB

        total_batch = 4
        strategy = self.scheduler.build_batch_strategy(
            KsanaModelKey.Wan2_2_T2V_14B, self.latent_shape, total_batch, torch.float16, self.device
        )

        # 内存充足时应该只有一个批次，包含所有样本
        self.assertEqual(len(strategy), 1)
        self.assertEqual(strategy[0].start, 0)
        self.assertEqual(strategy[0].end, 4)
        self.assertTrue(strategy[0].combine_cond_uncond)

    @patch("ksana.scheduler.scheduler.get_available_memory")
    def test_build_batch_strategy_limited_memory(self, mock_get_memory):
        """测试内存有限时的批处理策略"""
        # 模拟极度有限的内存，强制每个样本单独处理
        mock_get_memory.return_value = 100 * 1024 * 1024  # 100MB

        total_batch = 3
        strategy = self.scheduler.build_batch_strategy(
            KsanaModelKey.Wan2_2_T2V_14B, self.latent_shape, total_batch, torch.float16, self.device
        )

        # 内存不足时应该分成多个单样本批次
        self.assertEqual(len(strategy), 3)

        # 验证每个批次只处理一个样本且不合并CFG
        for i, item in enumerate(strategy):
            self.assertEqual(item.start, i)
            self.assertEqual(item.end, i + 1)
            self.assertFalse(item.combine_cond_uncond)


if __name__ == "__main__":
    unittest.main()
