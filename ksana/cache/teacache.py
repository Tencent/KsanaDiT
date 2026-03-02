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

import numpy as np
import torch

from ..config.cache_config import TeaCacheConfig
from ..models.model_key import KsanaModelKey
from ..utils import log
from ..utils.conf import load_cache_yaml_keys
from .base_cache import KsanaStepCache

TEACACHE_COEFFICIENTS, DEFAULT_THRESHOLDS = load_cache_yaml_keys(
    "teacache.yaml",
    ["TEACACHE_COEFFICIENTS", "DEFAULT_THRESHOLDS"],
)


class TeaCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: TeaCacheConfig):
        super().__init__(model_key, config)
        mode = config.mode if config.mode else "t2v"
        self.threshold = config.threshold if config.threshold is not None else DEFAULT_THRESHOLDS.get(mode, 0.2)
        self.cache_device = config.cache_device
        self.verbose = config.verbose if config.verbose is not None else False
        self.mode = mode
        self.start_step = config.start_step if config.start_step is not None else 0
        self.end_step = config.end_step

        self._coefficients = self._get_coefficients()
        self._rescale_func = np.poly1d(self._coefficients)

        self._reset_state()

        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0

    def _get_coefficients(self):
        if self.mode in ("t2v", "t2v_14B"):
            key = "t2v_14B"
        elif self.mode == "t2v_1.3B":
            key = "t2v_1.3B"
        elif self.mode in ("i2v", "i2v_720P"):
            key = "i2v_720P"
        elif self.mode == "i2v_480P":
            key = "i2v_480P"
        else:
            key = "t2v_14B"
            log.warning(f"[TeaCache] Unknown mode '{self.mode}', using default coefficients: {key}")

        coeffs = TEACACHE_COEFFICIENTS.get("WAN2.1", {})
        return coeffs.get(key, coeffs.get("t2v_14B", []))

    def _reset_state(self):
        self._residual_cache = [None, None]
        self._prev_e = [None, None]
        self._accumulated_error = [0.0, 0.0]
        self._should_calc = [True, True]
        self._ori_x = None
        self._call_count = 0
        self._generation_started = False

    def _get_branch(self, phase):
        if phase == "cond":
            return 0
        elif phase == "uncond":
            return 1
        return None

    def _make_decision(self, e, branch):
        if self._prev_e[branch] is None:
            self._should_calc[branch] = True
            self._prev_e[branch] = e.clone()
            return

        prev_e = self._prev_e[branch]
        prev_e_mean = prev_e.abs().mean()
        if prev_e_mean < 1e-8:
            self._should_calc[branch] = True
            self._prev_e[branch] = e.clone()
            return

        rel_l1 = ((e - prev_e).abs().mean() / prev_e_mean).cpu().item()
        scaled_dist = self._rescale_func(rel_l1)
        self._accumulated_error[branch] += scaled_dist

        if self.verbose:
            log.info(
                f"[TeaCache] branch={branch} rel_l1={rel_l1:.6f} "
                f"scaled={scaled_dist:.6f} acc_err={self._accumulated_error[branch]:.6f} "
            )

        if self._accumulated_error[branch] < self.threshold:
            self._should_calc[branch] = False
        else:
            self._should_calc[branch] = True
            self._accumulated_error[branch] = 0.0

        self._prev_e[branch] = e.clone()

    def _is_in_active_range(self, step_iter):
        if step_iter is None:
            return True
        if step_iter < self.start_step:
            return False
        if self.end_step is not None and step_iter >= self.end_step:
            return False
        return True

    def valid_for(self, phase=None, x=None, step_iter=None, timestep=None, e=None, **kwargs) -> bool:
        if step_iter is not None:
            if step_iter == 0 and self._generation_started:
                self._on_new_generation()
            self._generation_started = True

        self._total_calls += 1
        self._call_count += 1

        if not self._is_in_active_range(step_iter):
            self._cache_misses += 1
            return False

        if phase == "combine":
            return self._valid_for_combine(x, step_iter, e)

        branch = self._get_branch(phase)
        if branch is None:
            self._cache_misses += 1
            return False

        if self._residual_cache[branch] is None:
            self._cache_misses += 1
            return False

        # TeaCache needs e for cache hit validation
        if e is None:
            self._cache_misses += 1
            return False

        self._make_decision(e, branch)

        if not self._should_calc[branch]:
            self._cache_hits += 1
            return True
        else:
            self._cache_misses += 1
            return False

    def _valid_for_combine(self, x, step_iter, e):
        if self._residual_cache[0] is None or self._residual_cache[1] is None:
            self._cache_misses += 1
            return False

        if e is None:
            self._cache_misses += 1
            return False

        bsz = e.shape[0] // 2
        e_cond = e[:bsz]

        self._make_decision(e_cond, 0)

        if not self._should_calc[0]:
            self._cache_hits += 1
            return True
        else:
            self._cache_misses += 1
            return False

    def __call__(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs):
        # cond + uncond combined output
        if phase == "combine":
            r0 = self._residual_cache[0]
            r1 = self._residual_cache[1]
            if r0 is None or r1 is None:
                return None

            bsz = x.shape[0] // 2
            x_cond, x_uncond = x[:bsz], x[bsz:]

            if self.cache_device == "cpu":
                if r0.device.type == "cpu":
                    r0 = r0.to(x.device)
                if r1.device.type == "cpu":
                    r1 = r1.to(x.device)

            out_cond = x_cond + r0
            out_uncond = x_uncond + r1
            return torch.cat([out_cond, out_uncond], dim=0)

        branch = self._get_branch(phase)
        if branch is None:
            return x

        residual = self._residual_cache[branch]
        if residual is None:
            return None

        if self.cache_device == "cpu" and residual.device.type == "cpu":
            residual = residual.to(x.device)

        return x + residual

    def record_input_before_update(self, x=None, step_iter=None, timestep=None, **kwargs):
        self._ori_x = x.clone() if x is not None else None

    def update_cache(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs):
        if self._ori_x is None or x is None:
            return

        if phase == "combine":
            self._update_cache_combine(x, step_iter)
        else:
            branch = self._get_branch(phase)
            if branch is not None:
                self._update_cache_single(branch, x, step_iter)

        self._ori_x = None

    def _update_cache_combine(self, x, step_iter):
        bsz = x.shape[0] // 2
        out_cond, out_uncond = x[:bsz], x[bsz:]
        ori_cond, ori_uncond = self._ori_x[:bsz], self._ori_x[bsz:]

        for branch, (out, ori) in enumerate(zip([out_cond, out_uncond], [ori_cond, ori_uncond])):
            residual = out - ori
            if self.cache_device == "cpu":
                residual = residual.cpu()
            self._residual_cache[branch] = residual

    def _update_cache_single(self, branch, x, step_iter):
        residual = x - self._ori_x
        if self.cache_device == "cpu":
            residual = residual.cpu()
        self._residual_cache[branch] = residual

    def _on_new_generation(self):
        if self.verbose or self._total_calls > 0:
            self.show_cache_rate()
        self._reset_stats()
        self._reset_state()

    def _reset_stats(self):
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0

    def offload_to_cpu(self):
        for i in range(2):
            if self._residual_cache[i] is not None:
                self._residual_cache[i] = self._residual_cache[i].cpu()

    def show_cache_rate(self):
        if self._total_calls > 0:
            hit_rate = self._cache_hits / self._total_calls * 100
            miss_rate = self._cache_misses / self._total_calls * 100
            lines = [
                f"  Total calls:     {self._total_calls}",
                f"  Cache hits:      {self._cache_hits} ({hit_rate:.1f}%)",
                f"  Cache misses:    {self._cache_misses} ({miss_rate:.1f}%)",
            ]
            if self._cache_misses > 0:
                speedup = self._total_calls / self._cache_misses
                lines.append(f"  Estimated speedup: {speedup:.2f}x")
            log.info("TeaCache Generation Summary:\n%s", "\n".join(lines))
