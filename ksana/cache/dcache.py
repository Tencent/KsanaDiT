from .base_cache import KsanaStepCache
from ..config.cache_config import DCacheConfig
import numpy as np
import torch

from ..models.model_key import KsanaModelKey
from ..utils import log
from ..utils.torch_compile import disable_dynamo
from ..utils.utils import get_recommend_config

__all__ = ["DCache"]


DCACHE_COEFFS_MAPS = {
    KsanaModelKey.Wan2_2_T2V_14B_HIGH: [
        1.79787941e-06,
        -6.73669299e-03,
        9.45476817e00,
        -5.89111662e03,
        1.37508591e06,
    ],
    KsanaModelKey.Wan2_2_T2V_14B_LOW: [
        -4.67923490e-09,
        9.48489344e-06,
        -6.97084940e-03,
        2.12416998e00,
        -1.65571475e02,
    ],
}


RECOMMEND_DCACHE_CONFIGS = {
    KsanaModelKey.Wan2_2_T2V_14B_HIGH: DCacheConfig(
        name=KsanaModelKey.Wan2_2_T2V_14B_HIGH.name,
        fast_degree=70,
        slow_degree=35,
        fast_force_calc_every_n_step=1,
        slow_force_calc_every_n_step=5,
    ),
    KsanaModelKey.Wan2_2_T2V_14B_LOW: DCacheConfig(
        name=KsanaModelKey.Wan2_2_T2V_14B_LOW.name,
        fast_degree=65,
        slow_degree=25,
        fast_force_calc_every_n_step=2,
        slow_force_calc_every_n_step=4,
    ),
}


def _get_coeffs(model_key: KsanaModelKey):
    try:
        return DCACHE_COEFFS_MAPS[model_key]
    except KeyError:
        raise RuntimeError(f"dcache do not support model {model_key} yet")


class DCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: DCacheConfig):
        super().__init__(model_key, config)
        self.config = get_recommend_config(config, RECOMMEND_DCACHE_CONFIGS[model_key])
        self.degree_func = np.poly1d(_get_coeffs(model_key))
        self.need_compile_cache = False

        self.cur_input_x = None
        self.cur_degree = None
        self.prev_diff = {"cond": None, "uncond": None, "combine": None}
        self.cnt_continuous_cached = {"cond": 0, "uncond": 0, "combine": 0}
        self.total_in_cache_cnt = {"cond": 0, "uncond": 0, "combine": 0}
        self.total_cached_cnt = {"cond": 0, "uncond": 0, "combine": 0}
        self.total_cnt = {"cond": 0, "uncond": 0, "combine": 0}

    def __str__(self):
        return f"DitCache {self.model_key}, config: {self.config}"

    @staticmethod
    @disable_dynamo()
    def get_unify_degree(degree_func, timestep):
        # import ipdb
        # ipdb.set_trace()
        unify_degree = degree_func(timestep)
        unify_degree = abs(unify_degree)
        if hasattr(unify_degree, "__len__"):
            unify_degree = [min(unify_degree[i], 180 - unify_degree[i]) for i in range(len(unify_degree))]
        else:
            unify_degree = min(unify_degree, abs(180 - unify_degree))
        return unify_degree

    @disable_dynamo()
    def valid_for(self, phase: str, step_iter: int, timestep: int, **kwargs) -> bool:
        self.total_cnt[phase] += 1
        base_no_in_use_info = f"[MISS cache] phase {phase} step_iter {step_iter} timestep {timestep}"
        if self.total_cnt[phase] <= self.config.skip_first_n_iter:
            log.debug(
                f"{base_no_in_use_info}, by skip first {self.config.skip_first_n_iter} iter, current iter {self.total_cnt[phase]}"
            )
            return False
        if self.prev_diff[phase] is None:
            log.debug(f"{base_no_in_use_info}, by prev_diff is None")
            return False
        self.cur_degree = self.get_unify_degree(self.degree_func, timestep)
        if self.cur_degree > self.config.fast_degree:
            log.debug(f"{base_no_in_use_info}, by degree {self.cur_degree:.1f} > fast degree {self.config.fast_degree}")
            return False
        self.total_in_cache_cnt[phase] += 1
        if (
            self.config.slow_degree < self.cur_degree <= self.config.fast_degree
            and self.cnt_continuous_cached[phase] >= self.config.fast_force_calc_every_n_step
        ):
            log.debug(f"{base_no_in_use_info}, by every {self.config.fast_force_calc_every_n_step} steps")
            self.cnt_continuous_cached[phase] = 0
            return False
        if (
            self.cur_degree <= self.config.slow_degree
            and self.cnt_continuous_cached[phase] >= self.config.slow_force_calc_every_n_step
        ):
            log.debug(f"{base_no_in_use_info}, by every {self.config.slow_force_calc_every_n_step} steps")
            self.cnt_continuous_cached[phase] = 0
            return False
        return True

    @disable_dynamo()
    def __call__(self, phase: str, x: torch.Tensor, step_iter: int, timestep: int):
        base_info = f"phase {phase} step_iter {step_iter} timestep {timestep}"
        if self.prev_diff[phase] is None:
            log.error(f"prev_diff is None, call valid_for or update cache firstly for {base_info}")
            return None
        log.debug(f"[HIT cache] {base_info}, cur_degree {self.cur_degree:.1f}")
        self.total_cached_cnt[phase] += 1
        self.cnt_continuous_cached[phase] += 1
        cur_diff = self.prev_diff[phase]
        output = x + cur_diff.to(device=x.device, dtype=x.dtype)
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

    @disable_dynamo()
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
        cond_in_cache_total = self.total_in_cache_cnt["cond"]
        cond_cached_total = self.total_cached_cnt["cond"]
        cond_total = self.total_cnt["cond"]
        uncond_in_cache_total = self.total_in_cache_cnt["uncond"]
        uncond_cached_total = self.total_cached_cnt["uncond"]
        uncond_total = self.total_cnt["uncond"]
        log.info(
            f"DCache {self.model_key} fast degree {self.config.fast_degree}, slow degree {self.config.slow_degree}, "
            f"fast force compute every {self.config.fast_force_calc_every_n_step} steps in cache, "
            f"slow force compute every {self.config.slow_force_calc_every_n_step} steps in cache, "
            f"cond(in cache rate {(100 * cond_in_cache_total/ (cond_total + 1e-8)): .2f}%, "
            f"cached rate {(100 * cond_cached_total / (cond_total + 1e-8)): .2f}%), "
            f"uncond(in cache rate {(100 * uncond_in_cache_total / (uncond_total + 1e-8)): .2f}%, "
            f"cached rate {(100 * uncond_cached_total / (uncond_total + 1e-8)): .2f}%)"
        )
