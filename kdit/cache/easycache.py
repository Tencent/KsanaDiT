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

import torch

from ..config.cache_config import EasyCacheConfig
from ..models.model_key import KsanaModelKey
from ..utils import log
from .base_cache import KsanaStepCache

DEFAULT_THRESH = {
    "t2v": 0.06,
    "i2v": 0.05,
    "ti2v": 0.05,
}


class EasyCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: EasyCacheConfig):
        super().__init__(model_key, config)
        mode = config.mode if config.mode is not None else "t2v"
        default_thresh = DEFAULT_THRESH.get(mode, 0.05)
        self.threshold = config.reuse_thresh if config.reuse_thresh is not None else default_thresh
        self.start_percent = config.start_percent if config.start_percent is not None else 0.2
        self.end_percent = config.end_percent if config.end_percent is not None else 0.98
        self.cache_device = config.cache_device
        self.verbose = config.verbose if config.verbose is not None else False
        self.mode = mode

        self._reset_state()
        self._num_steps = None

        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0
        self._skip_by_retention = 0
        self._skip_by_no_cache = 0
        self._skip_by_not_ready = 0
        self._skip_by_error = 0

    def _reset_state(self):
        self._step_counter = 0

        self._residual_cache = [None, None]

        self._prev_input = None
        self._prev_output = None
        self._prev_prev_input = None
        self._k = None

        self._accumulated_error = 0.0
        self._should_calc_current_pair = True
        self._last_pred_change = None

        self._ori_x_full = None

    def setup(self, num_steps):
        self._num_steps = num_steps * 2  # CFG 导致步数翻倍
        self._reset_state()

    def _get_branch(self, phase):
        if phase == "cond":
            return 0
        elif phase == "uncond":
            return 1
        return None

    def _get_ret_steps(self):
        if self._num_steps is None:
            return 0
        return int(self._num_steps * self.start_percent)

    def _get_cutoff_steps(self):
        if self._num_steps is None:
            return float("inf")
        return int(self._num_steps * self.end_percent)

    def _is_in_retention_period(self, cnt):
        return cnt < self._get_ret_steps() or cnt >= self._get_cutoff_steps()

    def _make_decision(self, raw_input, cnt):
        if self._is_in_retention_period(cnt):
            self._should_calc_current_pair = True
            self._accumulated_error = 0.0
            self._prev_input = raw_input.clone()
            self._last_pred_change = None
            return

        if self._prev_input is None or self._prev_output is None or self._k is None:
            self._should_calc_current_pair = True
            self._prev_input = raw_input.clone()
            self._last_pred_change = None
            return

        input_change = (raw_input - self._prev_input).abs().mean().item()

        output_norm = self._prev_output.abs().mean().item()
        if output_norm > 1e-8:
            pred_change = self._k * (input_change / output_norm)
        else:
            pred_change = 0.0

        self._accumulated_error += pred_change
        self._last_pred_change = float(pred_change)

        acc_err_before_reset = self._accumulated_error
        if self._accumulated_error < self.threshold:
            self._should_calc_current_pair = False
        else:
            self._should_calc_current_pair = True
            self._accumulated_error = 0.0

        self._prev_input = raw_input.clone()
        if self.verbose:
            log.info(
                f"[EasyCache] cnt={cnt} input_change={input_change:.6f} output_norm={output_norm:.6f} "
                f"pred_change={self._last_pred_change:.6f} "
                f"acc_err={acc_err_before_reset:.6f} k={self._k:.6f} "
                f"thresh={self.threshold} decision={'CACHE' if not self._should_calc_current_pair else 'COMPUTE'}"
            )

    def _update_k_factor(self, output):
        if self._prev_output is None:
            return

        output_change = (output - self._prev_output).abs().mean()

        if self._prev_prev_input is not None and self._prev_input is not None:
            input_change = (self._prev_input - self._prev_prev_input).abs().mean()
            if input_change > 1e-8:
                self._k = (output_change / input_change).item()

        self._prev_prev_input = self._prev_input
        if self.verbose and self._k is not None:
            log.info(f"[EasyCache] update k={self._k:.6f}")

    def valid_for(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs) -> bool:
        self._total_calls += 1

        if phase == "combine":
            return self._valid_for_combine(x, step_iter)

        branch = self._get_branch(phase)
        if branch is None:
            return False

        cnt = step_iter * 2 + branch

        if self._is_in_retention_period(cnt):
            self._cache_misses += 1
            self._skip_by_retention += 1
            return False

        if self._residual_cache[branch] is None:
            self._cache_misses += 1
            self._skip_by_no_cache += 1
            return False

        if branch == 0:
            self._make_decision(x, cnt)

        if not self._should_calc_current_pair:
            self._cache_hits += 1
            return True
        else:
            self._cache_misses += 1
            if self._k is None or self._prev_output is None:
                self._skip_by_not_ready += 1
            else:
                self._skip_by_error += 1
            return False

    def _valid_for_combine(self, x, step_iter):
        cnt_cond = step_iter * 2
        cnt_uncond = step_iter * 2 + 1

        if self._is_in_retention_period(cnt_cond) or self._is_in_retention_period(cnt_uncond):
            self._cache_misses += 2
            self._skip_by_retention += 2
            return False

        if self._residual_cache[0] is None or self._residual_cache[1] is None:
            self._cache_misses += 2
            self._skip_by_no_cache += 2
            return False

        bsz = x.shape[0] // 2
        x_cond = x[:bsz]
        self._make_decision(x_cond, cnt_cond)

        if not self._should_calc_current_pair:
            self._cache_hits += 2
            return True
        else:
            self._cache_misses += 2
            if self._k is None or self._prev_output is None:
                self._skip_by_not_ready += 2
            else:
                self._skip_by_error += 2
            return False

    def __call__(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs):
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
        self._ori_x_full = x.clone() if x is not None else None

    def update_cache(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs):
        if self._ori_x_full is None or x is None:
            return

        if phase == "combine":
            self._update_cache_combine(x, step_iter)
        else:
            branch = self._get_branch(phase)
            if branch is not None:
                self._update_cache_single(branch, x, step_iter)

        self._ori_x_full = None

    def _update_cache_combine(self, x, step_iter):
        bsz = x.shape[0] // 2
        out_cond, out_uncond = x[:bsz], x[bsz:]
        ori_cond, ori_uncond = self._ori_x_full[:bsz], self._ori_x_full[bsz:]

        for branch, (out, ori) in enumerate(zip([out_cond, out_uncond], [ori_cond, ori_uncond])):
            residual = out - ori
            if self.cache_device == "cpu":
                residual = residual.cpu()
            self._residual_cache[branch] = residual

            if branch == 0:
                self._update_k_factor(out)
                self._prev_output = out.clone()

        self._accumulated_error = 0.0
        self._step_counter = max(self._step_counter, (step_iter + 1) * 2)
        self._check_generation_complete()

    def _update_cache_single(self, branch, x, step_iter):
        residual = x - self._ori_x_full
        if self.cache_device == "cpu":
            residual = residual.cpu()
        self._residual_cache[branch] = residual

        if branch == 0:
            self._update_k_factor(x)
            self._prev_output = x.clone()

        self._accumulated_error = 0.0
        cnt = step_iter * 2 + branch
        self._step_counter = max(self._step_counter, cnt + 1)
        self._check_generation_complete()

    def _check_generation_complete(self):
        if self._num_steps is not None and self._step_counter >= self._num_steps:
            self._on_generation_complete()

    def _on_generation_complete(self):
        if self.verbose:
            self.show_cache_rate()
        self._reset_stats()
        self._reset_state()

    def _reset_stats(self):
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0
        self._skip_by_retention = 0
        self._skip_by_no_cache = 0
        self._skip_by_not_ready = 0
        self._skip_by_error = 0

    def offload_to_cpu(self):
        for i in range(2):
            if self._residual_cache[i] is not None:
                self._residual_cache[i] = self._residual_cache[i].cpu()

    def show_cache_rate(self):
        if self._total_calls > 0:
            hit_rate = self._cache_hits / self._total_calls * 100
            skip_count = self._cache_hits
            compute_count = self._total_calls - self._cache_hits

            lines = [
                f"  Total calls:     {self._total_calls}",
                f"  Cache hits:      {skip_count} ({hit_rate:.1f}%)",
                f"  Full computes:   {compute_count} ({100-hit_rate:.1f}%)",
                "  Compute reasons breakdown:",
                f"    - Retention period: {self._skip_by_retention}",
                f"    - No cache:         {self._skip_by_no_cache}",
                f"    - Not ready(k):     {self._skip_by_not_ready}",
                f"    - Error threshold:  {self._skip_by_error}",
            ]
            if compute_count > 0:
                speedup = self._total_calls / compute_count
                lines.append(f"  Estimated speedup: {speedup:.2f}x")
            log.info("EasyCache Generation Summary:\n%s", "\n".join(lines))
