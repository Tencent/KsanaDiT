from .base_cache import KsanaCache
from .cache_config import DCacheConfig
import numpy as np

from ..utils import log

__all__ = ["DCache"]

try:
    # Avoid Dynamo compiling cache helpers that use numpy/Python control flow
    from torch._dynamo import disable as _dynamo_disable
except Exception:

    def _dynamo_disable(fn=None):
        return fn if fn is not None else (lambda f: f)


DCACHE_COEFFS_MAPS = {
    "wan2.2-high": {
        "t2v": {
            "A14B": [
                1.79787941e-06,
                -6.73669299e-03,
                9.45476817e00,
                -5.89111662e03,
                1.37508591e06,
            ]
        }
    },
    "wan2.2-low": {
        "t2v": {
            "A14B": [
                -4.67923490e-09,
                9.48489344e-06,
                -6.97084940e-03,
                2.12416998e00,
                -1.65571475e02,
            ]
        }
    },
}


def get_coeffs(
    model_name: str,
    model_type: str,
    model_size: str,
):
    try:
        coeffs = DCACHE_COEFFS_MAPS[model_name][model_type][model_size]
        return coeffs
    except KeyError:
        raise ValueError(f"Unknown model kind {model_name}, type {model_type}, size {model_size}")


class DCache(KsanaCache):
    def __init__(self, model_name: str, model_type: str, model_size: str, config: DCacheConfig):
        super().__init__(f"{model_name}_{model_type}_{model_size}_{config.name}")
        self.config = config
        coeffs = get_coeffs(model_name, model_type, model_size)
        self.degree_func = np.poly1d(coeffs)
        self.need_compile_cache = False

        self.prev_diff = {"cond": None, "uncond": None, "batch": None}
        self.cnt_continuous_cached = {"cond": 0, "uncond": 0, "batch": 0}
        self.total_in_cache_cnt = {"cond": 0, "uncond": 0, "batch": 0}
        self.total_cached_cnt = {"cond": 0, "uncond": 0, "batch": 0}
        self.total_cnt = {"cond": 0, "uncond": 0, "batch": 0}

    def __str__(self):
        return (
            f"DitCache {self.name} timestep_range: [{self.timestep_start}, {self.timestep_end}] "
            f"need_compile_cache: {self.need_compile_cache}, config: {self.config}"
        )

    # def __call__(self, *args, **kwds):
    #     return super().__call__(*args, **kwds)

    @staticmethod
    @_dynamo_disable()
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

    @_dynamo_disable()
    def can_use_cache(self, phase: str, current_x_input, current_timestep: int) -> bool:
        self.total_cnt[phase] += 1
        if self.total_cnt[phase] <= self.config.skip_first_n_iter:
            log.debug(
                f"[NOT use cache] phase {phase} timestep {current_timestep}, by skip first {self.config.skip_first_n_iter} iter, current iter {self.total_cnt[phase]}"
            )
            return False
        if self.prev_diff[phase] is None:
            log.debug(f"[NOT use cache] phase {phase} timestep {current_timestep}, by prev_diff is None")
            return False
        cur_degree = self.get_unify_degree(self.degree_func, current_timestep)
        self.cur_degree = cur_degree
        if cur_degree > self.config.fast_degree:
            log.debug(
                f"[NOT use cache] phase {phase} timestep {current_timestep}, by degree {cur_degree:.1f} > fast degree {self.config.fast_degree}"
            )
            return False
        else:
            return True

    @_dynamo_disable()
    def try_get_prev_cache(self, phase: str, current_x_input, current_timestep: int):
        self.total_in_cache_cnt[phase] += 1
        if (
            self.config.slow_degree < self.cur_degree <= self.config.fast_degree
            and self.cnt_continuous_cached[phase] >= self.config.fast_force_calc_every_n_step
        ):
            log.debug(
                f"[NOT use cache] phase {phase} timestep {current_timestep}, by every {self.config.fast_force_calc_every_n_step} steps"
            )
            self.cnt_continuous_cached[phase] = 0
            return None
        if (
            self.cur_degree <= self.config.slow_degree
            and self.cnt_continuous_cached[phase] >= self.config.slow_force_calc_every_n_step
        ):
            log.debug(
                f"[NOT use cache] phase {phase} timestep {current_timestep}, by every {self.config.slow_force_calc_every_n_step} steps"
            )
            self.cnt_continuous_cached[phase] = 0
            return None
        log.debug(f"[USE cache] phase {phase} timestep {current_timestep}, cur_degree {self.cur_degree:.1f}")
        cur_diff = self.prev_diff[phase]
        self.total_cached_cnt[phase] += 1
        self.cnt_continuous_cached[phase] += 1
        self.prev_diff[phase] = cur_diff
        return cur_diff

    @_dynamo_disable
    def offload_to_cpu(self):
        if self.prev_diff is not None:
            for phase in self.prev_diff:
                if self.prev_diff[phase] is not None:
                    self.prev_diff[phase] = self.prev_diff[phase].to("cpu")

    # def post_cacheprocess(self, phase: str, current_timestep: int, current_x_diff):
    #     self.prev_diff[phase] = current_x_diff

    @_dynamo_disable()
    def clone_input_x(self, current_timestep: int, current_x_input) -> bool:
        return current_x_input.clone().to("cpu") if self.config.offload else current_x_input.clone()

    @_dynamo_disable()
    def update_states(self, phase: str, current_timestep: int, current_x_input, current_x_output):
        current_x_output = current_x_output.clone().to("cpu") if self.config.offload else current_x_output
        self.prev_diff[phase] = current_x_output - current_x_input
        del current_x_input

    @_dynamo_disable()
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
            f"DCache {self.name} fast degree {self.config.fast_degree}, slow degree {self.config.slow_degree}, "
            f"fast force compute every {self.config.fast_force_calc_every_n_step} steps in cache, "
            f"slow force compute every {self.config.slow_force_calc_every_n_step} steps in cache, "
            f"cond(in cache rate {(100 * cond_in_cache_total/ (cond_total + 1e-8)): .2f}%, "
            f"cached rate {(100 * cond_cached_total / (cond_total + 1e-8)): .2f}%), "
            f"uncond(in cache rate {(100 * uncond_in_cache_total / (uncond_total + 1e-8)): .2f}%, "
            f"cached rate {(100 * uncond_cached_total / (uncond_total + 1e-8)): .2f}%)"
        )
