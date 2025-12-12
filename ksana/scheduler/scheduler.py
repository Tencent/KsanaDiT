import torch
from dataclasses import dataclass
from typing import List
from ..utils.memory import get_available_memory, estimate_ksana_model_memory, MODEL_SIZE_MAP, MEMORY_USAGE_FACTOR_MAP


@dataclass(frozen=True)
class BatchStrategyItem:
    start: int
    end: int
    combine_cond_uncond: bool


class KsanaScheduler:
    """
    Ksana调度器，负责批处理逻辑和内存管理
    """

    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config

    def _estimate_memory_for_batch(self, latent_shape, batch_size: int, run_dtype):
        shape = list(latent_shape)
        shape[0] = batch_size

        model_weight_memory = MODEL_SIZE_MAP.get(self.pipeline_config.model_size, 28 * 1024 * 1024 * 1024)  # 默认28GB

        memory_usage_factor = MEMORY_USAGE_FACTOR_MAP.get(
            (self.pipeline_config.model_name, self.pipeline_config.task_type, self.pipeline_config.model_size), 1.0
        )

        memory_required, minimum_required = estimate_ksana_model_memory(
            model_weight_memory, shape, run_dtype, memory_usage_factor
        )
        return memory_required, minimum_required

    def build_batch_strategy(self, latent_shape, total_batch: int, run_dtype, device: torch.device):
        """
        latent_shape: latent张量的形状, [bs, z_dim, f, h, w]
        """
        strategy: List[BatchStrategyItem] = []
        start = 0

        while start < total_batch:
            remaining = total_batch - start
            safe_memory = get_available_memory(device)
            chunk = remaining
            combine_cfg = True

            while chunk > 0:
                memory_double, _ = self._estimate_memory_for_batch(latent_shape, chunk, run_dtype)
                if memory_double <= safe_memory:
                    break
                chunk -= 1

            # 如果连单个样本都无法处理，强制处理但不合并CFG
            if chunk == 0:
                chunk = 1
                combine_cfg = False

            strategy.append(BatchStrategyItem(start, start + chunk, combine_cfg))
            start += chunk

        return strategy
