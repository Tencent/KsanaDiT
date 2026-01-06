import torch

from ..config.cache_config import CustomStepCacheConfig
from ..utils import log
from ..utils.torch_compile import disable_dynamo
from .base_cache import KsanaStepCache

__all__ = ["CustomStepCache"]


class CustomStepCache(KsanaStepCache):

    def __init__(self, model_key, config: CustomStepCacheConfig):
        super().__init__(model_key, config)
        self.config = config
        self.prev_diff = {"cond": None, "uncond": None, "combine": None}
        self.total_cached_cnt = {"cond": 0, "uncond": 0, "combine": 0}
        self.total_cnt = {"cond": 0, "uncond": 0, "combine": 0}
        self.cur_input_x = None

    def __str__(self):
        return f"CustomStepCache {self.model_key}, config: {self.config}"

    @disable_dynamo()
    def valid_for(self, phase: str, step_iter: int, timestep: int, **kwargs) -> bool:
        self.total_cnt[phase] += 1
        base_no_in_use_info = f"[MISS cache] phase {phase} step_iter {step_iter} timestep {timestep}"
        if self.prev_diff[phase] is None:
            log.debug(f"{base_no_in_use_info}, by prev_diff is None")
            return False
        if step_iter not in self.config.steps:
            log.debug(f"{base_no_in_use_info}, by step_iter not in {self.config.steps}")
            return False
        if self.prev_diff[phase] is None or step_iter not in self.config.steps:
            log.debug(f"{base_no_in_use_info}, by prev diff is None, or step_iter not in {self.config.steps}")
            return False

        idx = self.config.steps.index(step_iter)
        if idx >= len(self.config.scales):
            log.error(f"{base_no_in_use_info}, by idx {idx} out of scales {self.config.scales}")
            return False
        return True

    @disable_dynamo()
    def __call__(self, phase: str, x: torch.Tensor, step_iter: int, timestep: int, **kwargs) -> torch.Tensor:
        base_info = f"phase {phase} step_iter {step_iter} timestep {timestep}"
        log.debug(f"[HIT cache] {base_info}")
        if self.prev_diff[phase] is None:
            log.error(f"prev_diff is None, call valid_for or update cache firstly for {base_info}")
            return None
        self.total_cached_cnt[phase] += 1
        idx = self.config.steps.index(step_iter)
        cur_diff = self.prev_diff[phase]
        output = x + cur_diff.to(device=x.device, dtype=x.dtype) * self.config.scales[idx]
        return output

    @disable_dynamo()
    def record_input_before_update(self, x: torch.Tensor, **kwargs):
        self.cur_input_x = x.clone().to("cpu") if self.config.offload else x.clone()

    @disable_dynamo()
    def update_cache(self, phase: str, x: torch.Tensor, step_iter: int, timestep: int):
        current_x_output = x.clone().to("cpu") if self.config.offload else x
        if self.cur_input_x is None:
            base_info = f"phase {phase} step_iter {step_iter} timestep {timestep}"
            log.error(f"cur_input_x is None, call record_input_before_blocks firstly for {base_info}")
            return
        self.prev_diff[phase] = current_x_output - self.cur_input_x
        self.cur_input_x = None

    @disable_dynamo
    def offload_to_cpu(self):
        if self.prev_diff is None:
            return
        for phase in self.prev_diff:
            if self.prev_diff[phase] is not None:
                self.prev_diff[phase] = self.prev_diff[phase].to("cpu")

    @disable_dynamo()
    def show_cache_rate(self):
        if self.need_compile_cache:
            return
        self.offload_to_cpu()
        cond_cached_total = self.total_cached_cnt["cond"]
        cond_total = self.total_cnt["cond"]
        uncond_cached_total = self.total_cached_cnt["uncond"]
        uncond_total = self.total_cnt["uncond"]
        combine_cached_total = self.total_cached_cnt["combine"]
        combine_total = self.total_cnt["combine"]
        log.info(
            f"CustomStepCache {self.model_key} config: {self.config}, "
            f"cond(cached rate {(100 * cond_cached_total / (cond_total + 1e-8)): .2f}%), "
            f"uncond(cached rate {(100 * uncond_cached_total / (uncond_total + 1e-8)): .2f}%), "
            f"combine(cached rate {(100 * combine_cached_total / (combine_total + 1e-8)): .2f}%)"
        )
