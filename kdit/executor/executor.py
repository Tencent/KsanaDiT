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

import dataclasses
from abc import ABC
from functools import partial

import torch

from ..accelerator import platform
from ..config import DistributedConfig
from ..distributed import shard_model
from ..models.model_key import ModelKey
from ..models.model_pool import ModelPool
from ..nodes.core.device_context import NodeDeviceContext
from ..nodes.core.pin_hub import PinHub
from ..tensor import TensorPool
from ..utils import log
from ..utils.logger import reset_logging
from .distributed_group import DistributedGroupManager

if platform.is_npu():
    import torch_npu  # noqa: F401  # pylint: disable=unused-import
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  # pylint: disable=unused-import


class Executor(ABC):
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

        # DAG 模式：Node 实例缓存（按 node_id 缓存，避免重复创建）
        self._node_cache: dict[int, object] = {}

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

    # ── DAG Node 执行入口 ─────────────────────────────────────────────

    def _get_or_create_node(self, node_def):
        """根据 NodeDef 获取或创建 Node 实例（缓存在 _node_cache 中）。

        LoaderNode 通过 LoaderNodeFactory.create(model_key) 创建；
        InferNode 通过 InferNodeFactory.create(node_type, model_key) 创建。
        """
        from ..nodes.core.node_factory import InferNodeFactory, LoaderNodeFactory

        node_id = node_def.node_id
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        if node_def.is_loader:
            node = LoaderNodeFactory.create(node_def.model_key)
        else:
            node = InferNodeFactory.create(node_def.node_type, node_def.model_key)

        self._node_cache[node_id] = node
        return node

    def _build_pin_hub(self, node_def, pins_mapping) -> PinHub:
        """根据 node_def 和 pins_mapping 构建 PinHub。"""
        return PinHub(
            node_id=node_def.node_id,
            pins_mapping=pins_mapping,
            tensor_pool=self.tensor_pool,
            model_pool=self.model_pool,
        )

    def _inject_context_defaults(self, node_def, context):
        """注入 Executor 层的默认值到 context — DeviceInfo。"""
        if context is None:
            return context
        if context.device is None:
            context = dataclasses.replace(context, device=self.device_ctx)
        return context

    def run_loader_node(self, node_def, pins_mapping, context):
        """LoaderNode 执行入口 — 构建 PinHub 并执行 Node。

        自动注入 DeviceInfo / dist_config / shard_fn 到 context，
        构建 PinHub 绑定本地 pool，调用 node.run(pins, context=context)。

        Args:
            node_def: ``NodeDef`` — Loader 节点定义（is_loader=True）。
            pins_mapping: ``dict`` — 由 ``compute_pins_mapping()`` 生成的 pin 映射。
            context: ``NodeContext`` — metadata 中存放 build_loader_kwargs() 的结果。
        """
        from ..nodes.core.node_types import NodeDispatchPolicy

        context = self._inject_context_defaults(node_def, context)

        # 注入 dist_config / shard_fn 到 metadata
        if context is not None and context.metadata is not None:
            context.metadata.setdefault("dist_config", self.dist_config)
            context.metadata.setdefault("shard_fn", self.shard_fn)

        node = self._get_or_create_node(node_def)
        policy = node.dispatch_policy

        is_active = policy == NodeDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0
        if is_active:
            pins = self._build_pin_hub(node_def, pins_mapping)
            node.run(pins, context=context)

    def run_infer_node(self, node_def, pins_mapping, context):
        """InferNode 执行入口 — 构建 PinHub 并执行 Node。

        自动注入 DeviceInfo 到 context，构建 PinHub 绑定本地 pool，
        管理 pre/post tensor 同步。

        Args:
            node_def: ``NodeDef`` — Infer 节点定义（is_loader=False）。
            pins_mapping: ``dict`` — 由 ``compute_pins_mapping()`` 生成的 pin 映射。
            context: ``NodeContext`` — 可序列化的上下文（device 字段由此方法注入）。
        """
        from ..nodes.core.node_types import NodeDispatchPolicy

        context = self._inject_context_defaults(node_def, context)

        node = self._get_or_create_node(node_def)
        policy = node.dispatch_policy

        self._pre_sync_tensors(node, policy)

        is_active = policy == NodeDispatchPolicy.ALL_ALL_ALL or self.device_ctx.rank_id == 0
        if is_active:
            pins = self._build_pin_hub(node_def, pins_mapping)
            node.run(pins, context=context)

        self._post_sync_tensors(node, node_def, policy)

    def _pre_sync_tensors(self, node, policy):
        """执行前的 tensor 同步（预留接口）。

        未来可根据 policy 的输入维度自动 broadcast/gather 输入 tensor。
        """
        pass

    def _post_sync_tensors(self, node, node_def, policy):
        """执行后的 tensor 同步。

        当前实现：R0_R0_BCAST 时 broadcast output_tensor_pins 到所有卡。
        使用 TensorPoolKey(node_id, pin) 确保与 PinHub 写入的 key 一致。
        """
        from ..nodes.core.node_types import NodeDispatchPolicy
        from ..tensor.tensor_pool_key import TensorPoolKey

        if policy == NodeDispatchPolicy.R0_R0_BCAST and self.device_ctx.world_size > 1:
            pool_keys = [TensorPoolKey(node_def.node_id, pin) for pin in node.output_tensor_pins]
            self.dist_group.broadcast_tensors(
                tensor_pool=self.tensor_pool,
                keys=pool_keys,
                src_rank=0,
                device=self.device_ctx.device,
            )

    def put_tensors(self, tensors: dict):
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
