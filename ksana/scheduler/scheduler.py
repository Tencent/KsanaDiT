from dataclasses import dataclass
from typing import List

import torch

from ..utils.memory import MODEL_MEMORY_CONFIG, estimate_ksana_model_memory, get_available_memory


@dataclass(frozen=True)
class BatchStrategyItem:
    start: int
    end: int
    combine_cond_uncond: bool


class KsanaBatchScheduler:
    """
    KsanaBatchScheduler for dynamic batch scheduling
    """

    def __init__(self):
        pass

    def _estimate_memory_for_batch(self, model_key, latent_shape, batch_size: int, run_dtype):
        shape = list(latent_shape)
        shape[0] = batch_size

        memory_config = MODEL_MEMORY_CONFIG.get(model_key)
        if memory_config is None:
            raise ValueError(f"Unknown model key: {model_key}")

        model_weight_memory = memory_config["model_size"]
        memory_usage_factor = memory_config["usage_factor"]

        memory_required, minimum_required = estimate_ksana_model_memory(
            model_weight_memory, shape, run_dtype, memory_usage_factor
        )
        return memory_required, minimum_required

    def build_batch_strategy(self, model_key, latent_shape, total_batch: int, run_dtype, device: torch.device):
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
                memory_double, _ = self._estimate_memory_for_batch(model_key, latent_shape, chunk, run_dtype)
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
