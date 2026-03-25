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

import atexit
import functools
import threading

import ray
import torch.distributed as dist

from ..accelerator import platform
from ..config import DistributedConfig
from ..executor import Executor, RayExecutor
from ..utils import log
from ..utils.distribute import get_gpu_count, get_torchrun_env, is_launched_by_torchrun
from ..utils.profile import time_profile


def get_engine(*args, **kwargs):
    """
    Get the default engine instance (backward compatible).
    Delegates to Engine.get_default().
    """
    return Engine.get_default(*args, **kwargs)


def pop_keys_in_kwargs(to_be_removed_keys, kwargs):
    for key in to_be_removed_keys:
        if key in kwargs:
            kwargs.pop(key, None)
            log.debug(f"pop key {key} from kwargs")
    return kwargs


class Engine:
    """
    Ksana engine that manages executors for model loading and inference.

    Supports both singleton (via get_default()) and multi-instance (via direct __init__) usage.
    The get_engine() function delegates to get_default() for backward compatibility.
    """

    _default_instance = None  # class-level singleton cache
    _lock = threading.Lock()  # class-level lock for thread safety

    FUNC_KEY_PRE_ALL = "func_key_pre_all"
    FUNC_KEY_PRE_RAY = "func_key_pre_ray"
    FUNC_KEY_PRE_LOCAL = "func_key_pre_local"
    RAY_KEY_REMOVE_KWARGS = "ray_key_remove_kwargs"
    FUNC_KEY_POST_RAY_OUTPUTS = "func_key_post_ray_outputs"

    executors = None

    def __init__(
        self,
        dist_config: DistributedConfig = DistributedConfig(),
        offload_device="cpu",
        _register_atexit=False,
    ):
        """
        Initialize the Engine.

        Args:
            dist_config: Distributed configuration.
            offload_device: Device for offloading (default: "cpu").
            _register_atexit: Internal flag. When True, registers atexit cleanup.
                Only the default instance (created via get_default) should set this.
        """
        log.info(f"Initializing Engine with dist_config: {dist_config}, offload_device: {offload_device}")
        self.num_gpus = dist_config.num_gpus
        self._is_ray = False
        self._cleaned_up = False
        self.init_executors(dist_config=dist_config, offload_device=offload_device)
        if _register_atexit:
            atexit.register(self.cleanup_distributed)

    @classmethod
    def get_default(cls, *args, **kwargs) -> "Engine":
        """
        Get the default singleton instance (thread-safe).

        If the instance already exists and arguments are passed, a warning is logged
        and the arguments are ignored (existing instance is returned).
        """
        with cls._lock:
            if cls._default_instance is None:
                cls._default_instance = cls(*args, _register_atexit=True, **kwargs)
            elif args or kwargs:
                log.warning(
                    "Engine.get_default() called with arguments but instance already exists. "
                    "Arguments are ignored. Use Engine() to create a new instance."
                )
            return cls._default_instance

    @classmethod
    def reset_default(cls):
        """
        Reset the default instance (for testing / hot-reload).

        WARNING: Only call when no active inference is running.
        Code holding references to the old engine will break after reset.
        """
        with cls._lock:
            if cls._default_instance is not None:
                cls._default_instance.cleanup_distributed()
            cls._default_instance = None

    def init_executors(self, dist_config: DistributedConfig = None, offload_device=None):
        if dist_config.num_gpus == 1:
            self.executors = Executor(0, offload_device=offload_device)
            return
        if dist_config.num_gpus > get_gpu_count():
            raise ValueError(f"num_gpus({dist_config.num_gpus}) must be less than or equal to {get_gpu_count()}")

        if is_launched_by_torchrun():
            world_size, rank_id, local_rank_id, _ = get_torchrun_env()
            if world_size != dist_config.num_gpus:
                raise ValueError(f"world_size({world_size}) must be equal to num_gpus({dist_config.num_gpus})")
            self.executors = Executor(device_id=local_rank_id, offload_device=offload_device)
            self.executors.init_torch_dist_group(rank_id, dist_config=dist_config)
        else:
            # ray local device id always be 0
            local_rank_id = 0

            # TODO(rockcao): Refactor to unify NPU/GPU initialization logic (extract common code, use strategy pattern)
            # TODO(rockcao): Add Ray executor tests for distributed ops (all_gather，all_to_all)
            if platform.is_npu():
                from ray.util.placement_group import PlacementGroupSchedulingStrategy

                ray.init(resources={"NPU": dist_config.num_gpus})
                pg = ray.util.placement_group(
                    [{"NPU": 1.0} for _ in range(dist_config.num_gpus)],
                    strategy="PACK",
                )
                log.info("wait placement group ready")
                # Pre-populate bundle_cache to avoid a protobuf compatibility bug
                pg.bundle_cache = [{"NPU": 1.0} for _ in range(dist_config.num_gpus)]
                ray.get(pg.ready(), timeout=600)

                log.info(f"placement group is ready: {pg}")

                self.executors = []
                for i in range(dist_config.num_gpus):
                    strategy = PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    )
                    executor = RayExecutor.options(
                        scheduling_strategy=strategy,
                    ).remote(local_rank_id, offload_device)
                    self.executors.append(executor)
            else:
                ray.init(num_gpus=dist_config.num_gpus)
                self.executors = [
                    RayExecutor.remote(local_rank_id, offload_device) for _ in range(dist_config.num_gpus)
                ]
            init_futures = []
            # executors is sorted by rank_id
            for rank_id, executor in enumerate(self.executors):
                future = executor.init_torch_dist_group.remote(rank_id, dist_config)
                init_futures.append(future)
            ray.get(init_futures)
            self._is_ray = True

    @property
    def is_ray(self):
        return self._is_ray and ray.is_initialized()

    def _check_key_in_map(self, key: str, map: dict):
        return isinstance(map, dict) and map is not None and key in map and map[key] is not None

    def _check_callable_key_in_map(self, key: str, map: dict):
        return self._check_key_in_map(key, map) and callable(map[key])

    def _get_rank_0_result(self, func_res: list, *args, **kwargs):
        RANK_0_ID = 0  # pylint: disable=invalid-name
        return func_res[RANK_0_ID]

    # TODO: need remove or modify auto_dispatch
    @staticmethod
    def auto_dispatch(func):
        """auto dispatch the function to ray executors or local executor"""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = func.__name__  # 自动获取被装饰函数的名称
            pre_func_map = func(self, *args, **kwargs)
            if pre_func_map is not None and not isinstance(pre_func_map, dict):
                raise ValueError(
                    f"func{func} must return None or dict like:"
                    f'["{self.FUNC_KEY_PRE_RAY}":ray_pre_func, "{self.FUNC_KEY_PRE_LOCAL}":local_pre_func]'
                    f", but got {type(pre_func_map)}"
                )
            if self.executors is None:
                raise RuntimeError("executors is not initialized")

            if self._check_callable_key_in_map(self.FUNC_KEY_PRE_ALL, pre_func_map):
                pre_func_map[self.FUNC_KEY_PRE_ALL](*args, **kwargs)

            if self.is_ray:
                if self._check_callable_key_in_map(self.FUNC_KEY_PRE_RAY, pre_func_map):
                    pre_func_map[self.FUNC_KEY_PRE_RAY](*args, **kwargs)
                if self._check_key_in_map(self.RAY_KEY_REMOVE_KWARGS, pre_func_map):
                    to_be_remove = pre_func_map[self.RAY_KEY_REMOVE_KWARGS]
                    if not isinstance(to_be_remove, list):
                        to_be_remove = [to_be_remove]
                    kwargs = pop_keys_in_kwargs(to_be_remove, kwargs)
                func_futures = [getattr(executor, method_name).remote(*args, **kwargs) for executor in self.executors]
                # Note: the result is list by rank_id
                func_return = ray.get(func_futures)
                process_outputs_func = self._get_rank_0_result  # default get rank 0 result
                if self._check_callable_key_in_map(self.FUNC_KEY_POST_RAY_OUTPUTS, pre_func_map):
                    process_outputs_func = pre_func_map[self.FUNC_KEY_POST_RAY_OUTPUTS]
                func_return = process_outputs_func(func_return, *args, **kwargs)
                log.debug(f"method_name {method_name} final return: {func_return}")
                return func_return
            else:
                if self._check_callable_key_in_map(self.FUNC_KEY_PRE_LOCAL, pre_func_map):
                    pre_func_map[self.FUNC_KEY_PRE_LOCAL](*args, **kwargs)
                executor_func = getattr(self.executors, method_name)
                if executor_func is None:
                    raise ValueError(f"method_name {method_name} not found in executors")
                func_return = executor_func(*args, **kwargs)
                log.debug(f"method_name {method_name} single result: {func_return}")
                return func_return

        return wrapper

    # ── @auto_dispatch 方法：通过 Executor 同名方法分发 ─────────────────────

    @auto_dispatch
    def clear_models(self, *args, **kwargs):
        pass

    # ── V5 Node 架构：统一入口 ──────────────────────────────────────────

    def run_node(self, node_def, input_pins, context) -> dict:
        """统一 Node 执行入口 — 根据 node_def 类型自动生成 profile label 并分发到所有 Executor。

        Args:
            node_def: ``NodeDef`` — 节点定义。
            input_pins: ``dict`` — 由 ``compute_input_pins()`` 生成的 pin 映射。
            context: ``NodeContext`` — 可序列化的上下文。

        Returns:
            output_pins — ``{TensorKey | ModelKey: TensorPoolKey | ModelPoolKey}`` 映射。
            Ray 模式下取 rank 0 的结果（output_pins 是纯元数据，所有 rank 相同）。
        """
        node_label = self._build_node_label(node_def)

        with time_profile(node_label):
            if self.is_ray:
                results = ray.get([ex.run_node.remote(node_def, input_pins, context) for ex in self.executors])
                return results[0]
            else:
                return self.executors.run_node(node_def, input_pins, context)

    @staticmethod
    def _build_node_label(node_def) -> str:
        """根据 NodeDef 生成 time_profile 标签。"""
        if node_def.is_io:
            # IO (Loader) — "load_[MODEL_NAME]" 或 "load_node_{id}"
            if node_def.model_key is not None:
                model_name = node_def.model_key.name if hasattr(node_def.model_key, "name") else str(node_def.model_key)
                return f"load_[{model_name}]"
            return f"load_node_{node_def.node_id}"

        # Infer — "NODE_TYPE[MODEL_NAME]" 或 "NODE_TYPE" 或 "node_{id}"
        label = node_def.node_type.name if node_def.node_type is not None else f"node_{node_def.node_id}"
        if node_def.model_key is not None:
            model_name = node_def.model_key.name if hasattr(node_def.model_key, "name") else str(node_def.model_key)
            label = f"{label}[{model_name}]"
        return label

    def clear_all_tensors(self):
        """清理所有 Executor 的 tensor_pool — 用于 Pipeline/ComfyUI 的 try/finally 异常恢复。"""
        self._clear_tensor_pools()

    def register_tensor(self, pool_key, ref_count):
        """注册 tensor 引用计数 — 透传到所有 Executor 的 tensor_pool。

        Args:
            pool_key: ``TensorPoolKey`` — tensor 的 pool key。
            ref_count: ``int`` — 下游消费者数量。
        """
        if self.is_ray:
            ray.get([ex.register_ref_count.remote(pool_key, ref_count) for ex in self.executors])
        else:
            self.executors.register_ref_count(pool_key, ref_count)

    def _clear_tensor_pools(self, exclude=None):
        """清理所有 Executor 的 tensor pool。"""
        if self.is_ray:
            ray.get([ex.clear_tensor_pool.remote(exclude=exclude) for ex in self.executors])
        else:
            self.executors.clear_tensor_pool(exclude=exclude)

    def put_tensors(self, tensors: dict):
        """将 tensor 写入所有 Executor 的 tensor_pool。

        用于外部（如 ComfyUI adapter）向 Node 传递输入数据。

        Args:
            tensors: ``{TensorKey: tensor}`` 映射，value 为 None 的条目会被跳过。
        """
        tensors = {k: v for k, v in tensors.items() if v is not None}
        if not tensors:
            return
        if self.is_ray:
            futures = [ex.put_tensors.remote(tensors) for ex in self.executors]
            ray.get(futures)
        else:
            for key, tensor in tensors.items():
                self.executors.tensor_pool.put(key, tensor)

    def get_tensor(self, key):
        """从 rank 0 Executor 的 tensor_pool 读取 TensorValue。

        所有最终输出都在 rank 0 上，自动从 rank 0 取，无需指定 rank。
        返回 ``TensorValue``，调用方通过 ``.data`` 获取裸 tensor。
        """
        if self.is_ray:
            return ray.get(self.executors[0].get_tensor.remote(key))
        return self.executors.tensor_pool.get(key)

    def has_tensor(self, key):
        """检查 rank 0 Executor 的 tensor_pool 中是否存在指定 key。"""
        if self.is_ray:
            return ray.get(self.executors[0].has_tensor.remote(key))
        return self.executors.tensor_pool.has(key)

    def rename_tensor(self, old_key, new_key):
        """重命名所有 Executor 的 tensor_pool 中的 key。

        用于 Pipeline 编排时在两个 run_node 之间调整 key 名称，
        使上游 Node 的输出 key 匹配下游 Node 的输入 key。
        """
        if self.is_ray:
            ray.get([ex.rename_tensor.remote(old_key, new_key) for ex in self.executors])
        else:
            self.executors.rename_tensor(old_key, new_key)

    # ── 清理 ───────────────────────────────────────────────────────────

    def cleanup_distributed(self):
        """Tear down dist process-group and Ray runtime. Idempotent — safe to call multiple times."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        # 清理模型和显存（必须在 ray.shutdown 之前，否则 Ray actor 已销毁无法调用）
        try:
            self.clear_models()
        except Exception:  # pylint: disable=broad-except
            log.warning("clear_models failed during cleanup_distributed", exc_info=True)

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if self.is_ray:
            ray.shutdown()
