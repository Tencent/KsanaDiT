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

from abc import ABC
from functools import partial

import torch

from ..accelerator import platform
from ..config import DistributedConfig
from ..distributed import shard_model
from ..models.model_key import ModelKey
from ..models.model_pool import ModelPool
from ..nodes.core.device_context import NodeDeviceContext
from ..tensor import TensorPool
from ..utils import log
from ..utils.logger import reset_logging
from .distributed_group import DistributedGroupManager

if platform.is_npu():
    import torch_npu  # noqa: F401  # pylint: disable=unused-import
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  # pylint: disable=unused-import


class KsanaExecutor(ABC):
    """
    Base class for all Ksana executors.
    和模型有关的配置信息不放在Executor中，而是放在ModelBase中
    这里只放和device，分布式相关的信息
    """

    def __init__(self, device_id: int = 0, offload_device: str = "cpu"):
        """
        Initialize the executor.
        """
        self.device_id = device_id
        self.rank_id = device_id
        self.world_size = 1
        self.device = torch.device(f"cuda:{self.device_id}")
        self.offload_device = torch.device(offload_device)
        torch.cuda.set_device(self.device)
        # Note: each executor has its own model pool for nodes call, and pipeline own engine then can use executors
        self.model_pool = ModelPool()
        self.shard_fn = None
        self.dist_config = DistributedConfig(num_gpus=1, use_sp=False, dit_fsdp=False, ulysses_size=1)

        # V5 Node 架构：三大管理器
        self.tensor_pool = TensorPool()
        self.dist_group = DistributedGroupManager()
        self.device_ctx = self._build_device_ctx(self.device, self.offload_device, self.rank_id, self.world_size)

        log.info(f"create executor with device_id {self.device_id}, offload_device {self.offload_device}")
        reset_logging()

    def init_torch_dist_group(self, rank_id, dist_config: DistributedConfig):
        """r initialize sequence parallel group."""
        self.dist_config = dist_config
        log.info(f"init torch dist group with dist_config {dist_config}")
        if dist_config.num_gpus <= 1:
            return
        self.rank_id = rank_id
        self.world_size = dist_config.num_gpus
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if platform.is_gpu() else "hccl",
                init_method="env://",
                rank=rank_id,
                device_id=self.device,
                world_size=dist_config.num_gpus,
            )
        log.info(f"init distributed group with rank_id {self.rank_id}, world_size {self.world_size}")
        reset_logging(rank_id)
        self.shard_fn = partial(shard_model, device_id=self.device_id) if self.dist_config.dit_fsdp else None

        # V5: 同步 dist_group + 重建 device_ctx
        self.dist_group.init(rank_id, self.world_size)
        self.device_ctx = self._build_device_ctx(self.device, self.offload_device, rank_id, self.world_size)

    def clear_models(self, model_keys: list[ModelKey] | ModelKey = None):
        if self.model_pool is None:
            return
        self.model_pool.clear_models(model_keys)
        # 全量清理时，tensor_pool 中的 tensor 也不再有意义，一并清理
        if model_keys is None:
            self.tensor_pool.clear()

    # ── V5 Node 架构：统一入口 ──────────────────────────────────────────

    @staticmethod
    def _build_device_ctx(device, offload_device, rank_id, world_size) -> NodeDeviceContext:
        """构建 KsanaDeviceContext。"""
        return NodeDeviceContext(device=device, offload_device=offload_device, rank_id=rank_id, world_size=world_size)

    def run_loader_node(self, model_key, **kwargs):
        """统一的模型加载入口 — 根据 NodeDispatchPolicy 决定是否执行。

        自动注入 Executor 级别的 dist_config 和 shard_fn（Node 无需关心来源）。
        """
        from ..nodes.core.node_factory import LoaderNodeFactory
        from ..nodes.core.node_types import NodeDispatchPolicy

        kwargs.setdefault("dist_config", self.dist_config)
        kwargs.setdefault("shard_fn", self.shard_fn)

        node = LoaderNodeFactory.create(model_key)
        policy = node.dispatch_policy

        if policy == NodeDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0:
            node.run(model_key, model_pool=self.model_pool, device_ctx=self.device_ctx, **kwargs)

    def run_infer_node(self, infer_node_type, model_key, context):
        """统一的前向推理入口 — 根据 NodeDispatchPolicy 决定执行 + 同步。

        结果写入 tensor_pool，不返回值。外部通过 engine.get_tensor() 获取输出。
        """
        from ..nodes.core.node_factory import InferNodeFactory
        from ..nodes.core.node_types import NodeDispatchPolicy

        node = InferNodeFactory.create(infer_node_type, model_key)
        policy = node.dispatch_policy

        # TODO: 根据 policy 的输入维度，executor 在执行前自动管理输入 tensor 的同步
        # 例如 R0 输入 → 自动 broadcast input_tensor_keys 到所有卡
        # 这样 Node 内部不需要感知多卡，executor 负责 tensor 的 pre-sync 和 post-sync
        self._pre_sync_tensors(node, policy)

        is_active_rank = policy == NodeDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0
        if is_active_rank:
            node.run(
                model_key,
                context,
                tensor_pool=self.tensor_pool,
                model_pool=self.model_pool,
                device_ctx=self.device_ctx,
            )

        self._post_sync_tensors(node, policy)

    def _pre_sync_tensors(self, node, policy):
        """执行前的 tensor 同步（预留接口）。

        未来可根据 policy 的输入维度自动 broadcast/gather 输入 tensor。
        """
        pass

    def _post_sync_tensors(self, node, policy):
        """执行后的 tensor 同步。

        当前实现：R0_R0_BCAST 时 broadcast output_tensor_keys 到所有卡。
        """
        from ..nodes.core.node_types import NodeDispatchPolicy

        if policy == NodeDispatchPolicy.R0_R0_BCAST and self.device_ctx.world_size > 1:
            self.dist_group.broadcast_tensors(
                tensor_pool=self.tensor_pool,
                keys=node.output_tensor_keys,
                src_rank=0,
                device=self.device_ctx.device,
            )

    def put_tensors(self, **tensors):
        """将 tensor 写入 tensor_pool（由 Engine 桥接方法通过 Ray 调用）。"""
        for key, tensor in tensors.items():
            if tensor is not None:
                self.tensor_pool.put(key, tensor)

    def get_tensor(self, key):
        """从 tensor_pool 读取 TensorValue（由 Engine 桥接方法通过 Ray 调用）。"""
        return self.tensor_pool.get(key)

    def has_tensor(self, key):
        """检查 tensor_pool 中是否存在指定 key（由 Engine 桥接方法通过 Ray 调用）。"""
        return self.tensor_pool.has(key)

    def rename_tensor(self, old_key, new_key):
        """重命名 tensor_pool 中的 key（由 Engine 桥接方法通过 Ray 调用）。"""
        self.tensor_pool.rename(old_key, new_key)

    def clear_tensor_pool(self, exclude=None):
        """清理 tensor pool（session 结束时由 Engine 调用）。

        Args:
            exclude: 需要保留的 ``TensorKey`` 列表，不会被 release。
        """
        self.tensor_pool.clear(exclude=exclude)
