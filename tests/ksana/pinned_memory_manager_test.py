import unittest

import torch

from ksana.memory import PinnedMemoryManager


class TestPinnedMemoryManager(unittest.TestCase):
    def setUp(self):
        """每个测试前创建新的 manager 实例"""
        self.manager = PinnedMemoryManager()

    def test_basic_allocation(self):
        """测试基本的内存分配"""
        manager = self.manager

        # 分配 2GB（应该分配 2 个 1GB 的块）
        size_bytes = int(2 * 1024**3)
        blocks = manager.allocate_blocks(size_bytes)

        self.assertEqual(len(blocks), 2)
        for block in blocks:
            self.assertTrue(block.is_allocated)
            self.assertEqual(block.size_bytes, manager.default_block_size_bytes)

    def test_block_release(self):
        """测试内存块释放"""
        manager = self.manager

        # 分配
        size_bytes = int(3 * 1024**3)
        blocks = manager.allocate_blocks(size_bytes)
        self.assertEqual(len(blocks), 3)

        # 释放
        manager.release_blocks(blocks)
        for block in blocks:
            self.assertFalse(block.is_allocated)

    def test_block_reuse(self):
        """测试内存块复用"""
        manager = self.manager

        # 第一次分配
        size_bytes = int(5 * 1024**3)
        blocks1 = manager.allocate_blocks(size_bytes)
        block_ids_1 = {block.block_id for block in blocks1}

        # 释放
        manager.release_blocks(blocks1)

        # 第二次分配相同大小（应该复用）
        blocks2 = manager.allocate_blocks(size_bytes)
        block_ids_2 = {block.block_id for block in blocks2}

        self.assertEqual(block_ids_1, block_ids_2)

    def test_cross_dtype_reuse(self):
        """测试跨 dtype 的内存块复用"""
        manager = self.manager

        # 分配 3GB
        size_bytes = int(3 * 1024**3)
        blocks_fp32 = manager.allocate_blocks(size_bytes)
        block_ids_fp32 = {block.block_id for block in blocks_fp32}

        # 释放
        manager.release_blocks(blocks_fp32)

        # 再次分配相同大小（用于不同 dtype）
        blocks_fp16 = manager.allocate_blocks(size_bytes)
        block_ids_fp16 = {block.block_id for block in blocks_fp16}

        # 应该复用相同的块（跨 dtype）
        self.assertEqual(block_ids_fp32, block_ids_fp16)

    def test_partial_reuse(self):
        """测试部分复用场景"""
        manager = self.manager

        # 第一次分配 5GB
        blocks1 = manager.allocate_blocks(int(5 * 1024**3))
        self.assertEqual(len(blocks1), 5)

        # 释放
        manager.release_blocks(blocks1)

        # 第二次分配 8GB（应该复用 5 个，新建 3 个）
        blocks2 = manager.allocate_blocks(int(8 * 1024**3))
        self.assertEqual(len(blocks2), 8)

        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 8)
        self.assertEqual(stats["allocated_blocks"], 8)

    def test_concurrent_allocation(self):
        """测试并发分配"""
        manager = self.manager

        # 模型1 分配 4GB
        blocks_model1 = manager.allocate_blocks(int(4 * 1024**3))
        self.assertEqual(len(blocks_model1), 4)

        # 模型2 分配 3GB
        blocks_model2 = manager.allocate_blocks(int(3 * 1024**3))
        self.assertEqual(len(blocks_model2), 3)

        # 验证没有重叠
        ids_model1 = {block.block_id for block in blocks_model1}
        ids_model2 = {block.block_id for block in blocks_model2}
        self.assertEqual(len(ids_model1 & ids_model2), 0)

        # 统计信息
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 7)
        self.assertEqual(stats["allocated_blocks"], 7)
        self.assertEqual(stats["free_blocks"], 0)

    def test_get_stats(self):
        """测试统计信息"""
        manager = self.manager

        # 初始状态
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 0)
        self.assertEqual(stats["allocated_blocks"], 0)
        self.assertEqual(stats["free_blocks"], 0)

        # 分配 5GB
        blocks = manager.allocate_blocks(int(5 * 1024**3))
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 5)
        self.assertEqual(stats["allocated_blocks"], 5)
        self.assertEqual(stats["free_blocks"], 0)
        self.assertAlmostEqual(stats["total_memory_gb"], 5.0, places=1)
        self.assertAlmostEqual(stats["allocated_memory_gb"], 5.0, places=1)

        # 释放 3 个块
        manager.release_blocks(blocks[:3])
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 5)
        self.assertEqual(stats["allocated_blocks"], 2)
        self.assertEqual(stats["free_blocks"], 3)
        self.assertAlmostEqual(stats["allocated_memory_gb"], 2.0, places=1)
        self.assertAlmostEqual(stats["free_memory_gb"], 3.0, places=1)

    def test_block_properties(self):
        """测试 block 的属性"""
        manager = self.manager

        blocks = manager.allocate_blocks(int(2 * 1024**3))
        block = blocks[0]

        # 验证 block 属性
        self.assertIsInstance(block.block_id, int)
        self.assertEqual(block.size_bytes, int(1 * 1024**3))
        self.assertIsInstance(block.buffer, torch.Tensor)
        self.assertEqual(block.buffer.dtype, torch.uint8)
        self.assertTrue(block.buffer.is_pinned())
        self.assertTrue(block.is_allocated)

    def test_dtype_agnostic_buffer(self):
        """测试 dtype 无关的 buffer"""
        manager = self.manager

        blocks = manager.allocate_blocks(int(1 * 1024**3))
        block = blocks[0]

        # 验证可以转换为不同的 dtype
        buffer_fp32 = block.buffer[: 4 * 100].view(torch.float32).view(100)
        self.assertEqual(buffer_fp32.shape, (100,))
        self.assertEqual(buffer_fp32.dtype, torch.float32)

        buffer_fp16 = block.buffer[: 2 * 100].view(torch.float16).view(100)
        self.assertEqual(buffer_fp16.shape, (100,))
        self.assertEqual(buffer_fp16.dtype, torch.float16)

        buffer_int8 = block.buffer[:100].view(torch.int8).view(100)
        self.assertEqual(buffer_int8.shape, (100,))
        self.assertEqual(buffer_int8.dtype, torch.int8)

    def test_clear_all(self):
        """测试清空所有块"""
        manager = self.manager

        # 分配一些块
        manager.allocate_blocks(int(5 * 1024**3))
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 5)

        # 清空
        manager.clear_all()
        stats = manager.get_stats()
        self.assertEqual(stats["total_blocks"], 0)

    def test_custom_block_size(self):
        """测试自定义块大小"""
        manager = self.manager

        # 使用自定义块大小（2GB）
        custom_block_size = int(2 * 1024**3)
        blocks = manager.allocate_blocks(int(6 * 1024**3), block_size_bytes=custom_block_size)

        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertEqual(block.size_bytes, custom_block_size)


if __name__ == "__main__":
    unittest.main()
