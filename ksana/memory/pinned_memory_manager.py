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

"""
Pinned Memory Manager - 管理固定大小的 pinned memory 块

该管理器用于避免重复申请大块 pinned memory，通过预分配固定大小的块（默认 1GB）
并在多个模型之间复用这些内存块。
"""

import threading

import torch

from ksana.utils.logger import log


class PinnedMemoryBlock:
    """表示一个固定大小的 pinned memory 块"""

    def __init__(self, block_id: int, size_bytes: int):
        """
        初始化一个 pinned memory 块

        Args:
            block_id: 块的唯一标识符
            size_bytes: 块的大小（字节）
        """
        self.block_id = block_id
        self.size_bytes = size_bytes

        # 使用 uint8 作为底层存储，这样可以被任意 dtype 复用
        # 分配 pinned memory
        self.buffer = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
        self.is_allocated = False  # 是否已被分配给某个模型

        log.debug(f"Created PinnedMemoryBlock {block_id}: " f"{size_bytes / 1024**3:.2f} GB (dtype-agnostic)")

    def mark_allocated(self):
        """标记该块已被分配"""
        self.is_allocated = True

    def mark_free(self):
        """标记该块已被释放"""
        self.is_allocated = False


class PinnedMemoryManager:
    """
    Pinned Memory Manager

    管理固定大小的 pinned memory 块，支持多个模型复用内存。
    """

    def __init__(self):
        self._blocks: list[PinnedMemoryBlock] = []  # 所有 blocks 不再按 dtype 分组
        self._block_counter = 0
        self._allocation_lock = threading.Lock()

        # 默认块大小：1GB
        self.default_block_size_gb = 1
        self.default_block_size_bytes = int(self.default_block_size_gb * 1024**3)

        log.info(f"PinnedMemoryManager initialized with default block size: " f"{self.default_block_size_gb} GB")

    def allocate_blocks(self, total_size_bytes: int, block_size_bytes: int = None) -> list[PinnedMemoryBlock]:
        """
        为模型分配所需的 pinned memory 块

        Args:
            total_size_bytes: 总共需要的内存大小（字节）
            block_size_bytes: 单个块的大小（字节），如果为 None 则使用默认值

        Returns:
            分配的内存块列表
        """
        if block_size_bytes is None:
            block_size_bytes = self.default_block_size_bytes

        # 计算需要多少个块
        num_blocks_needed = (total_size_bytes + block_size_bytes - 1) // block_size_bytes

        log.debug(f"Allocating {num_blocks_needed} blocks for {total_size_bytes / 1024**3:.2f} GB")

        with self._allocation_lock:
            allocated_blocks = []
            num_reused = 0
            num_new = 0

            # 首先尝试复用已有的空闲块（只检查大小，不检查 dtype）
            for block in self._blocks:
                if not block.is_allocated and block.size_bytes == block_size_bytes:
                    block.mark_allocated()
                    allocated_blocks.append(block)
                    num_reused += 1
                    if len(allocated_blocks) == num_blocks_needed:
                        break

            # 如果空闲块不够，创建新的块
            while len(allocated_blocks) < num_blocks_needed:
                new_block = PinnedMemoryBlock(block_id=self._block_counter, size_bytes=block_size_bytes)
                self._block_counter += 1
                new_block.mark_allocated()
                self._blocks.append(new_block)
                allocated_blocks.append(new_block)
                num_new += 1

            log.debug(f"Allocated {len(allocated_blocks)} blocks " f"(reused: {num_reused}, new: {num_new})")

            return allocated_blocks

    def release_blocks(self, blocks: list[PinnedMemoryBlock]):
        """
        释放内存块，使其可以被其他模型复用

        Args:
            blocks: 要释放的内存块列表
        """
        with self._allocation_lock:
            for block in blocks:
                block.mark_free()

            log.info(f"Released {len(blocks)} blocks")

    def get_stats(self) -> dict:
        """
        获取内存管理器的统计信息

        Returns:
            包含统计信息的字典
        """
        with self._allocation_lock:
            stats = {
                "total_blocks": len(self._blocks),
                "allocated_blocks": 0,
                "free_blocks": 0,
                "total_memory_gb": 0.0,
                "allocated_memory_gb": 0.0,
                "free_memory_gb": 0.0,
            }

            for block in self._blocks:
                size_gb = block.size_bytes / 1024**3
                stats["total_memory_gb"] += size_gb

                if block.is_allocated:
                    stats["allocated_blocks"] += 1
                    stats["allocated_memory_gb"] += size_gb
                else:
                    stats["free_blocks"] += 1
                    stats["free_memory_gb"] += size_gb

            return stats

    def clear_all(self):
        """
        清空所有内存块（谨慎使用！）
        """
        with self._allocation_lock:
            self._blocks.clear()
            self._block_counter = 0
            log.warning("All pinned memory blocks have been cleared")
