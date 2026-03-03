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
from ..config import KsanaDistributedConfig
from ..executor import KsanaExecutor, RayKsanaExecutor
from ..utils import log
from ..utils.distribute import get_gpu_count, get_torchrun_env, is_launched_by_torchrun


def get_engine(*args, **kwargs):
    """
    Get the default engine instance (backward compatible).
    Delegates to KsanaEngine.get_default().
    """
    return KsanaEngine.get_default(*args, **kwargs)


def pop_keys_in_kwargs(to_be_removed_keys, kwargs):
    for key in to_be_removed_keys:
        if key in kwargs:
            kwargs.pop(key, None)
            log.debug(f"pop key {key} from kwargs")
    return kwargs


class KsanaEngine:
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
        dist_config: KsanaDistributedConfig = KsanaDistributedConfig(),
        offload_device="cpu",
        _register_atexit=False,
    ):
        """
        Initialize the KsanaEngine.

        Args:
            dist_config: Distributed configuration.
            offload_device: Device for offloading (default: "cpu").
            _register_atexit: Internal flag. When True, registers atexit cleanup.
                Only the default instance (created via get_default) should set this.
        """
        log.info(f"Initializing KsanaEngine with dist_config: {dist_config}, offload_device: {offload_device}")
        self.num_gpus = dist_config.num_gpus
        self._is_ray = False
        self._cleaned_up = False
        self.init_executors(dist_config=dist_config, offload_device=offload_device)
        if _register_atexit:
            atexit.register(self.cleanup_distributed)

    @classmethod
    def get_default(cls, *args, **kwargs) -> "KsanaEngine":
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
                    "KsanaEngine.get_default() called with arguments but instance already exists. "
                    "Arguments are ignored. Use KsanaEngine() to create a new instance."
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

    def init_executors(self, dist_config: KsanaDistributedConfig = None, offload_device=None):
        if dist_config.num_gpus == 1:
            self.executors = KsanaExecutor(0, offload_device=offload_device)
            return
        if dist_config.num_gpus > get_gpu_count():
            raise ValueError(f"num_gpus({dist_config.num_gpus}) must be less than or equal to {get_gpu_count()}")

        if is_launched_by_torchrun():
            world_size, rank_id, local_rank_id, _ = get_torchrun_env()
            if world_size != dist_config.num_gpus:
                raise ValueError(f"world_size({world_size}) must be equal to num_gpus({dist_config.num_gpus})")
            self.executors = KsanaExecutor(device_id=local_rank_id, offload_device=offload_device)
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
                try:
                    ray.get(pg.ready(), timeout=600)
                except AttributeError:
                    # Pre-populate bundle_cache to avoid a protobuf compatibility bug
                    # in some Ray versions where _get_bundle_cache() fails with
                    # "AttributeError: 'str' object has no attribute 'DESCRIPTOR'".
                    # bundle_cache must be a list of resource dicts so that Ray's
                    # _validate_resource_shape iterates over the dicts, not integer keys.
                    if pg.bundle_cache is None:
                        pg.bundle_cache = [{"NPU": 1.0} for _ in range(dist_config.num_gpus)]
                log.info(f"placement group is ready: {pg}")

                self.executors = []
                for i in range(dist_config.num_gpus):
                    strategy = PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    )
                    executor = RayKsanaExecutor.options(
                        scheduling_strategy=strategy,
                    ).remote(local_rank_id, offload_device)
                    self.executors.append(executor)
            else:
                ray.init(num_gpus=dist_config.num_gpus)
                self.executors = [
                    RayKsanaExecutor.remote(local_rank_id, offload_device) for _ in range(dist_config.num_gpus)
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

    @auto_dispatch
    def load_models(self, *args, **kwargs):
        pass

    @auto_dispatch
    def clear_models(self, *args, **kwargs):
        pass

    @auto_dispatch
    def load_diffusion_model(self, *args, **kwargs):
        return {self.RAY_KEY_REMOVE_KWARGS: "comfy_bar_callback"}

    @auto_dispatch
    def load_text_encoder(self, *args, **kwargs):
        pass

    @auto_dispatch
    def load_vae_model(self, *args, **kwargs):
        pass

    @auto_dispatch
    def forward_generator(self, *args, **kwargs):
        return {self.RAY_KEY_REMOVE_KWARGS: "comfy_bar_callback"}

    @auto_dispatch
    def forward_vae_encode(self, *args, **kwargs):
        pass

    @auto_dispatch
    def forward_vae_encode_image(self, *args, **kwargs):
        pass

    @auto_dispatch
    def forward_vae_decode(self, *args, **kwargs):
        pass

    def broadcast_input_args(self, prompts, *, seed, prompts_negative=None, **kwargs):
        if dist.is_initialized():
            dist.broadcast_object_list([prompts, seed], src=0)
            if prompts_negative is not None:
                dist.broadcast_object_list(prompts_negative, src=0)

    @auto_dispatch
    def generate(self, *args, **kwargs):
        def pre_func_all(*args, **kwargs):
            # Note: here covers ray and gpus
            if self.num_gpus > 1:
                runtime_config = kwargs.get("runtime_config", None)
                seed = runtime_config.seed if runtime_config else None
                self.broadcast_input_args(*args, seed=seed, **kwargs)

        return {self.FUNC_KEY_PRE_ALL: pre_func_all, self.FUNC_KEY_POST_RAY_OUTPUTS: self._get_rank_0_result}

    def cleanup_distributed(self):
        """Tear down dist process-group and Ray runtime. Idempotent — safe to call multiple times."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        if self.is_ray:
            ray.shutdown()
