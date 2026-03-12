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

from __future__ import annotations

import torch
import torch.distributed as dist

from ..tensor import TensorPool
from ..utils.logger import log


class DistributedGroupManager:
    """管理 torch.distributed 进程组，提供 tensor broadcast 能力。

    与 TensorPool 配合：broadcast 时自动将 tensor 写入非 src_rank 的 pool。
    """

    def __init__(self):
        self.rank_id: int = 0
        self.world_size: int = 1
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init(self, rank_id: int, world_size: int) -> None:
        """初始化分布式信息（进程组由 Executor.init_torch_dist_group 负责创建）。"""
        self.rank_id = rank_id
        self.world_size = world_size
        self._initialized = world_size > 1 and dist.is_initialized()
        if self._initialized:
            log.info(f"DistributedGroupManager initialized: rank={rank_id}, world_size={world_size}")

    def broadcast_tensors(
        self,
        tensor_pool: TensorPool,
        keys: list[str],
        src_rank: int = 0,
        device: torch.device | None = None,
    ) -> None:
        """通过 NCCL/HCCL broadcast tensor 到所有卡。

        src_rank 的 tensor 从 tensor_pool 读取；
        非 src_rank 先接收 meta（shape + dtype），创建空 tensor，再接收数据并写入 tensor_pool。

        支持 ``list[torch.Tensor]``：先广播 list 长度，再逐个广播每个 tensor。
        """
        if not self._initialized or self.world_size <= 1:
            return

        for key in keys:
            if self.rank_id == src_rank:
                tensor_value = tensor_pool.get(key)
                if tensor_value is None:
                    log.debug(f"broadcast_tensors: key '{key}' not found on src_rank={src_rank}, skipping")
                    dist.broadcast_object_list([True], src=src_rank)
                    continue
                dist.broadcast_object_list([False], src=src_rank)

                value = tensor_value.data  # 取裸 tensor / list[Tensor]
                if isinstance(value, list):
                    self._broadcast_tensor_list(value, src_rank, device, is_src=True)
                    # src_rank 不需要重新写入 pool
                else:
                    self._broadcast_single_tensor(value, src_rank, device, is_src=True)
            else:
                skip_list: list[bool] = [False]
                dist.broadcast_object_list(skip_list, src=src_rank)
                if skip_list[0]:
                    continue

                # 先探测是 list 还是单 tensor
                is_list_flag: list[bool] = [False]
                dist.broadcast_object_list(is_list_flag, src=src_rank)

                if is_list_flag[0]:
                    recv_value = self._broadcast_tensor_list(None, src_rank, device, is_src=False)
                else:
                    recv_value = self._broadcast_single_tensor(None, src_rank, device, is_src=False)
                tensor_pool.put(key, recv_value)

    # ── 内部广播辅助方法 ──────────────────────────────────────────────────

    def _broadcast_single_tensor(
        self,
        tensor: torch.Tensor | None,
        src_rank: int,
        device: torch.device | None,
        *,
        is_src: bool,
    ) -> torch.Tensor | None:
        """广播单个 tensor。src_rank 发送 is_list=False + meta + data；非 src_rank 接收。"""
        if is_src:
            # 发送 is_list 标志
            dist.broadcast_object_list([False], src=src_rank)
            meta = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            dist.broadcast_object_list([meta], src=src_rank)
            tensor_on_device = tensor.to(device) if device and tensor.device != device else tensor
            dist.broadcast(tensor_on_device, src=src_rank)
            return None
        else:
            meta_list: list[dict | None] = [None]
            dist.broadcast_object_list(meta_list, src=src_rank)
            meta = meta_list[0]
            dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
            recv_tensor = torch.empty(meta["shape"], dtype=dtype, device=device)
            dist.broadcast(recv_tensor, src=src_rank)
            return recv_tensor

    def _broadcast_tensor_list(
        self,
        tensor_list: list[torch.Tensor] | None,
        src_rank: int,
        device: torch.device | None,
        *,
        is_src: bool,
    ) -> list[torch.Tensor] | None:
        """广播 list[Tensor]。先广播 is_list=True + 长度，再逐个广播每个 tensor。"""
        if is_src:
            dist.broadcast_object_list([True], src=src_rank)
            length = len(tensor_list)
            dist.broadcast_object_list([length], src=src_rank)
            for t in tensor_list:
                meta = {"shape": list(t.shape), "dtype": str(t.dtype)}
                dist.broadcast_object_list([meta], src=src_rank)
                t_on_device = t.to(device) if device and t.device != device else t
                dist.broadcast(t_on_device, src=src_rank)
            return None
        else:
            length_list: list[int] = [0]
            dist.broadcast_object_list(length_list, src=src_rank)
            length = length_list[0]
            results = []
            for _ in range(length):
                meta_list: list[dict | None] = [None]
                dist.broadcast_object_list(meta_list, src=src_rank)
                meta = meta_list[0]
                dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
                recv_tensor = torch.empty(meta["shape"], dtype=dtype, device=device)
                dist.broadcast(recv_tensor, src=src_rank)
                results.append(recv_tensor)
            return results

    def barrier(self) -> None:
        """同步屏障。"""
        if self._initialized:
            dist.barrier()
