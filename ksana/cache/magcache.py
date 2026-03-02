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

from ..config.cache_config import MagCacheConfig
from ..models.model_key import KsanaModelKey
from ..utils.conf import load_cache_yaml_keys_safe, save_cache_yaml_key
from ..utils.logger import log
from .base_cache import KsanaStepCache

_MAGCACHE_YAML = "magcache.yaml"
_MAGCACHE_ROOT_KEY = "WAN_MAG_RATIOS"

WAN_MAG_RATIOS: dict = load_cache_yaml_keys_safe(_MAGCACHE_YAML, [_MAGCACHE_ROOT_KEY])[0]


def _nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])
    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


class MagCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: MagCacheConfig):
        super().__init__(model_key, config)
        self.threshold = config.threshold if config.threshold is not None else 0.04
        max_skip = config.max_skip_steps if config.max_skip_steps is not None else config.k
        self.k = max_skip if max_skip is not None else 2
        self.retention_ratio = config.retention_ratio if config.retention_ratio is not None else 0.2
        self.cache_device = config.cache_device
        self.verbose = config.verbose if config.verbose is not None else False
        self.start_step = config.start_step
        self.end_step = config.end_step
        self._reset_state()

        self._mag_ratios = None
        self._num_steps = None
        self._mode = config.mode if config.mode is not None else "t2v"
        self._split_step = config.split_step
        self._config_mag_ratios = config.mag_ratios

        self._calibration_mode = False
        self._calibration_ratios: list[float] = []
        self._calibration_residual_cache: list[torch.Tensor | None] = [None, None]

        self._lazy_setup_defaults()

        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0
        self._skipped_by_retention = 0
        self._skipped_by_err = 0
        self._skipped_by_k = 0
        self._step_details = []

    def _reset_state(self):
        self._residual_cache = [None, None]
        self._accumulated_err = [0.0, 0.0]
        self._accumulated_steps = [0, 0]
        self._accumulated_ratio = [1.0, 1.0]
        self._ori_x = None
        self._step_counter = 0
        self._step_details = []

    def _has_ratios_available(self) -> bool:
        if self._config_mag_ratios is not None:
            return True
        return isinstance(WAN_MAG_RATIOS, dict) and (self._mode in WAN_MAG_RATIOS)

    def _lazy_setup_defaults(self):
        if self._mag_ratios is not None and self._num_steps is not None:
            return
        if self._config_mag_ratios is not None:
            base_ratios = np.array([1.0] * 2 + self._config_mag_ratios)
            self._mag_ratios = base_ratios
            self._num_steps = len(base_ratios)
        elif isinstance(WAN_MAG_RATIOS, dict) and self._mode in WAN_MAG_RATIOS:
            base_ratios = np.array([1.0] * 2 + WAN_MAG_RATIOS[self._mode])
            self._mag_ratios = base_ratios
            self._num_steps = len(base_ratios)
        else:
            self._calibration_mode = True
            log.info(
                "MagCache: no mag_ratios found for mode '%s', entering auto-calibration mode. "
                "The first generation will run without caching to collect ratios.",
                self._mode,
            )

    def setup(self, num_steps, mode=None, split_step=None, mag_ratios=None):
        self._num_steps = num_steps
        if mode is not None:
            self._mode = mode
        if split_step is not None:
            self._split_step = split_step
        self._reset_state()

        if mag_ratios is not None:
            base_ratios = np.array([1.0] * 2 + mag_ratios)
        elif self._config_mag_ratios is not None:
            base_ratios = np.array([1.0] * 2 + self._config_mag_ratios)
        elif isinstance(WAN_MAG_RATIOS, dict) and self._mode in WAN_MAG_RATIOS:
            base_ratios = np.array([1.0] * 2 + WAN_MAG_RATIOS[self._mode])
        else:
            self._calibration_mode = True
            self._calibration_ratios = []
            self._calibration_residual_cache = [None, None]
            log.info(
                "MagCache: no mag_ratios for mode '%s', calibration will run this generation.",
                self._mode,
            )
            return

        self._calibration_mode = False
        if len(base_ratios) != num_steps:
            half_steps = num_steps // 2
            ratio_con = _nearest_interp(base_ratios[0::2], half_steps)
            ratio_ucon = _nearest_interp(base_ratios[1::2], half_steps)
            self._mag_ratios = np.concatenate([ratio_con.reshape(-1, 1), ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        else:
            self._mag_ratios = base_ratios

    def _should_use_cache(self, cnt) -> bool:
        if self._calibration_mode:
            return False
        self._lazy_setup_defaults()
        if self._mag_ratios is None or self._num_steps is None:
            return False

        if self._split_step is not None:
            if self._mode == "i2v":
                threshold_step = int(self._split_step + (self._num_steps - self._split_step) * self.retention_ratio)
                if cnt < threshold_step:
                    return False
            else:  # t2v
                early_threshold = int(self._split_step * self.retention_ratio)
                late_threshold = int((self._num_steps - self._split_step) * self.retention_ratio + self._split_step)
                if cnt < early_threshold or (self._split_step <= cnt <= late_threshold):
                    return False
        else:  # ti2v and single-model cases
            if cnt < int(self._num_steps * self.retention_ratio):
                return False

        return True

    def _phase_cnt_pairs(self, phase, step_iter):
        if step_iter is None:
            return []
        if phase == "combine":
            base = int(step_iter) * 2
            return [(0, base), (1, base + 1)]
        if phase == "uncond":
            return [(1, int(step_iter) * 2 + 1)]
        # default: "cond" or unknown
        return [(0, int(step_iter) * 2)]

    def valid_for(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs) -> bool:
        self._total_calls += 1
        pairs = self._phase_cnt_pairs(phase, step_iter)
        if not pairs:
            return False

        if self._calibration_mode:
            return False

        self._lazy_setup_defaults()

        if phase == "combine":
            step_info = {"cnt": int(step_iter), "branch": "combine", "action": "compute", "reason": ""}
        else:
            b, cnt = pairs[0]
            step_info = {"cnt": cnt, "branch": "cond" if b == 0 else "uncond", "action": "compute", "reason": ""}

        if not self._should_use_cache(max(c for _, c in pairs)):
            self._skipped_by_retention += 1
            step_info["reason"] = "retention_period"
            self._step_details.append(step_info)
            return False

        for branch, cnt in pairs:
            if self._residual_cache[branch] is None:
                self._cache_misses += 1
                step_info["reason"] = "no_cache"
                self._step_details.append(step_info)
                return False
            if cnt >= len(self._mag_ratios):
                self._cache_misses += 1
                step_info["reason"] = "out_of_range"
                self._step_details.append(step_info)
                return False

            cur_mag_ratio = self._mag_ratios[cnt]
            new_ratio = self._accumulated_ratio[branch] * cur_mag_ratio
            new_steps = self._accumulated_steps[branch] + 1
            new_err = self._accumulated_err[branch] + abs(1 - new_ratio)
            if not (new_err < self.threshold and new_steps <= self.k):
                self._cache_misses += 1
                if not (new_err < self.threshold):
                    self._skipped_by_err += 1
                    step_info["reason"] = f"err_exceed({new_err:.4f}>={self.threshold})"
                else:
                    self._skipped_by_k += 1
                    step_info["reason"] = f"k_exceed({new_steps}>{self.k})"
                step_info["acc_err"] = round(new_err, 5)
                step_info["acc_steps"] = new_steps
                self._step_details.append(step_info)
                return False

        self._cache_hits += 1
        step_info["action"] = "skip"
        step_info["reason"] = "cache_hit"
        self._step_details.append(step_info)
        return True

    def __call__(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs) -> torch.Tensor:
        self._lazy_setup_defaults()
        pairs = self._phase_cnt_pairs(phase, step_iter)
        if not pairs:
            return x

        if phase == "combine":
            bsz = x.shape[0] // 2
            x0, x1 = x[:bsz], x[bsz:]
            for branch, cnt in pairs:
                cur_mag_ratio = self._mag_ratios[cnt]
                self._accumulated_ratio[branch] *= cur_mag_ratio
                self._accumulated_steps[branch] += 1
                self._accumulated_err[branch] += abs(1 - self._accumulated_ratio[branch])
            r0, r1 = self._residual_cache[0], self._residual_cache[1]
            if r0 is None or r1 is None:
                return None
            if self.cache_device == "cpu":
                if r0.device.type == "cpu":
                    r0 = r0.to(x.device)
                if r1.device.type == "cpu":
                    r1 = r1.to(x.device)
            out = torch.cat([x0 + r0, x1 + r1], dim=0)
            self._step_counter = max(self._step_counter, pairs[-1][1] + 1)
            if self._num_steps is not None and self._step_counter >= self._num_steps:
                self._on_generation_complete()
            return out

        branch, cnt = pairs[0]
        cur_mag_ratio = self._mag_ratios[cnt]
        self._accumulated_ratio[branch] *= cur_mag_ratio
        self._accumulated_steps[branch] += 1
        self._accumulated_err[branch] += abs(1 - self._accumulated_ratio[branch])

        residual = self._residual_cache[branch]
        if residual is None:
            return None

        if self.cache_device == "cpu" and residual.device.type == "cpu":
            residual = residual.to(x.device)

        out = x + residual
        self._step_counter = max(self._step_counter, cnt + 1)
        if self._num_steps is not None and self._step_counter >= self._num_steps:
            self._on_generation_complete()
        return out

    def record_input_before_update(self, x=None, step_iter=None, timestep=None, **kwargs):
        self._ori_x = x.clone() if x is not None else None

    def update_cache(self, phase=None, x=None, step_iter=None, timestep=None, **kwargs):
        self._lazy_setup_defaults()
        pairs = self._phase_cnt_pairs(phase, step_iter)
        if not pairs:
            self._ori_x = None
            return

        if self._ori_x is not None and x is not None:
            residual = x - self._ori_x

            if self._calibration_mode:
                self._collect_calibration_ratios(residual, phase, pairs)

            if phase == "combine":
                bsz = residual.shape[0] // 2
                r0, r1 = residual[:bsz], residual[bsz:]
                if self.cache_device == "cpu":
                    r0, r1 = r0.cpu(), r1.cpu()
                self._residual_cache[0] = r0
                self._residual_cache[1] = r1
            else:
                branch, _ = pairs[0]
                if self.cache_device == "cpu":
                    residual = residual.cpu()
                self._residual_cache[branch] = residual

        for branch, cnt in pairs:
            self._accumulated_err[branch] = 0.0
            self._accumulated_steps[branch] = 0
            self._accumulated_ratio[branch] = 1.0
            self._step_counter = max(self._step_counter, cnt + 1)

        if self._num_steps is not None and self._step_counter >= self._num_steps:
            self._on_generation_complete()

        self._ori_x = None

    def _collect_calibration_ratios(self, residual: torch.Tensor, phase, pairs):
        if phase == "combine":
            bsz = residual.shape[0] // 2
            branch_residuals = {0: residual[:bsz], 1: residual[bsz:]}
        else:
            branch, _ = pairs[0]
            branch_residuals = {branch: residual}

        for branch, cur_r in branch_residuals.items():
            prev_r = self._calibration_residual_cache[branch]
            if prev_r is not None:
                with torch.no_grad():
                    if prev_r.device != cur_r.device:
                        prev_r = prev_r.to(cur_r.device)
                    prev_norm = prev_r.norm(dim=-1)
                    prev_norm = prev_norm.clamp(min=1e-8)
                    ratio = (cur_r.norm(dim=-1) / prev_norm).mean().item()
                    self._calibration_ratios.append(round(ratio, 5))
            self._calibration_residual_cache[branch] = cur_r.detach().clone()

    def _finalize_calibration(self):
        ratios = self._calibration_ratios
        if not ratios:
            log.warning("MagCache calibration: no ratios collected, staying in calibration mode.")
            return

        self._mag_ratios = np.array([1.0] * 2 + ratios)
        self._num_steps = len(self._mag_ratios)
        self._calibration_mode = False
        self._calibration_residual_cache = [None, None]

        try:
            is_distributed = False
            try:
                import torch.distributed as dist

                is_distributed = dist.is_initialized()
            except (ImportError, RuntimeError):
                log.warning("MagCache calibration: torch.distributed not initialized")
            should_save = (not is_distributed) or (dist.get_rank() == 0)
            if should_save:
                yaml_key = f"{_MAGCACHE_ROOT_KEY}.{self._mode}"
                save_cache_yaml_key(_MAGCACHE_YAML, yaml_key, ratios)
        except (OSError, ValueError) as exc:
            log.warning("MagCache calibration: failed to save ratios to yaml: %s", exc)

        if isinstance(WAN_MAG_RATIOS, dict):
            WAN_MAG_RATIOS[self._mode] = ratios

        log.info(
            "MagCache Auto-Calibration Complete\n"
            "  Mode: %s\n"
            "  Collected %d ratio values\n"
            "  Total mag_ratios length (with padding): %d\n"
            "  Saved to settings/cache/%s under key '%s.%s'\n"
            "  Subsequent generations will use cached ratios.",
            self._mode,
            len(ratios),
            len(self._mag_ratios),
            _MAGCACHE_YAML,
            _MAGCACHE_ROOT_KEY,
            self._mode,
        )

    def _on_generation_complete(self):
        if self._calibration_mode:
            self._finalize_calibration()
        if self.verbose:
            self.show_cache_rate()
        self._reset_stats()
        self._reset_state()

    def _reset_stats(self):
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0
        self._skipped_by_retention = 0
        self._skipped_by_err = 0
        self._skipped_by_k = 0

    def offload_to_cpu(self):
        for i in range(2):
            if self._residual_cache[i] is not None:
                self._residual_cache[i] = self._residual_cache[i].cpu()

    def show_cache_rate(self):
        if self._total_calls > 0:
            hit_rate = self._cache_hits / self._total_calls * 100
            skip_count = self._cache_hits
            compute_count = self._total_calls - self._cache_hits
            other_count = compute_count - self._skipped_by_retention - self._skipped_by_err - self._skipped_by_k
            lines = [
                f"  Total steps:     {self._total_calls}",
                f"  Cache hits:      {skip_count} ({hit_rate:.1f}%)",
                f"  Full computes:   {compute_count} ({100-hit_rate:.1f}%)",
                "  Compute reasons breakdown:",
                f"    - Retention period: {self._skipped_by_retention}",
                f"    - Error threshold:  {self._skipped_by_err}",
                f"    - K limit:          {self._skipped_by_k}",
                f"    - No cache/other:   {other_count}",
                f"  Estimated speedup: {self._total_calls / max(compute_count, 1):.2f}x",
            ]
            log.info("MagCache Generation Summary:\n%s", "\n".join(lines))

    def get_step_details(self):
        return self._step_details
