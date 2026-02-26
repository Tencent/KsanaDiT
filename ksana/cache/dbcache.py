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

"""
DBCache (Dual Block Cache) for KsanaDiT.

This is a deep integration of cache-dit's DBCache algorithm into KsanaDiT,

DBCache leverages the feature similarity between adjacent timesteps in the
diffusion process. By computing L1 diff of residuals after the first _f_n blocks,
it decides whether to reuse cached residuals from previous steps or compute
all blocks.

Architecture:
    ┌──────────────┬─────────────────────────┬───────────────────┐
    │   _f_n Blocks  │      _m_n Blocks          │    _b_n Blocks      │
    │   (前n个)    │   (中间blocks)          │    (后n个)        │
    │   总是计算    │   可能被跳过/缓存        │    总是计算        │
    │   计算稳定diff│   复用cached residual   │    修正近似误差    │
    └──────────────┴─────────────────────────┴───────────────────┘

Reference:
    https://github.com/vipshop/cache-dit
"""

from dataclasses import dataclass, field

import torch

from ..config.cache_config import DBCacheConfig
from ..utils import log
from .base_cache import KsanaBlockCache

__all__ = ["DBCache", "DBCacheContext"]

from ..models.model_key import KsanaModelKey
from ..utils.torch_compile import disable_dynamo
from ..utils.types import evolve_with_recommend

RECOMMEND_DBCACHE_CONFIGS = {
    KsanaModelKey.Wan2_2_T2V_14B: DBCacheConfig(
        name=KsanaModelKey.Wan2_2_T2V_14B.name,
        fn_compute_blocks=1,
        bn_compute_blocks=0,
        residual_diff_threshold=0.08,
        max_warmup_steps=4,
        warmup_interval=1,
        max_cached_steps=8,
        max_continuous_cached_steps=2,
        enable_separate_cfg=True,
        num_blocks=40,
    ),
    # TODO(TJ): add high and low config
}


class DBCacheContext:
    def __init__(self, config: DBCacheConfig, num_blocks: int):
        self.config = config
        self.num_blocks = num_blocks

        self.current_step = 0
        self.cached_steps_count = 0
        self.continuous_cached_steps = 0
        self.last_advanced_timestep: int | None = None

        self.buffers: dict[str, CacheBuffer] = {
            "cond": CacheBuffer(),
            "uncond": CacheBuffer(),
            "combine": CacheBuffer(),
        }

        self.total_steps = {"cond": 0, "uncond": 0, "combine": 0}
        self.cached_steps = {"cond": 0, "uncond": 0, "combine": 0}

    def reset(self):
        self.current_step = 0
        self.cached_steps_count = 0
        self.continuous_cached_steps = 0
        for buffer in self.buffers.values():
            buffer.clear()
        self.total_steps = {"cond": 0, "uncond": 0}
        self.cached_steps = {"cond": 0, "uncond": 0}

    def mark_step_begin(self, phase: str = "cond"):
        self.total_steps[phase] += 1

    def is_warmup_step(self) -> bool:
        if self.current_step < self.config.max_warmup_steps:
            if self.config.warmup_interval > 1:
                return (self.current_step % self.config.warmup_interval) == 0
            return True
        return False

    def should_force_compute(self) -> bool:
        if self.config.max_continuous_cached_steps > 0:
            return self.continuous_cached_steps >= self.config.max_continuous_cached_steps
        return False

    def has_exceeded_max_cached_steps(self) -> bool:
        if self.config.max_cached_steps > 0:
            return self.cached_steps_count >= self.config.max_cached_steps
        return False

    @property
    def fn_blocks_range(self) -> tuple[int, int]:
        return (0, self.config.fn_compute_blocks)

    @property
    def mn_blocks_range(self) -> tuple[int, int]:
        _f_n = self.config.fn_compute_blocks
        _b_n = self.config.bn_compute_blocks
        if _b_n == 0:
            return (_f_n, self.num_blocks)
        return (_f_n, self.num_blocks - _b_n)

    @property
    def bn_blocks_range(self) -> tuple[int, int]:
        _b_n = self.config.bn_compute_blocks
        if _b_n == 0:
            return (self.num_blocks, self.num_blocks)
        return (self.num_blocks - _b_n, self.num_blocks)


@dataclass
class CacheBuffer:
    f_n_residual: torch.Tensor | None = None
    b_n_residual: torch.Tensor | None = None
    b_n_encoder_residual: torch.Tensor | None = None

    prev_fn_residuals: list = field(default_factory=list)
    prev_bn_residuals: list = field(default_factory=list)

    def clear(self):
        self.f_n_residual = None
        self.b_n_residual = None
        self.b_n_encoder_residual = None
        self.prev_fn_residuals.clear()
        self.prev_bn_residuals.clear()


class DBCache(KsanaBlockCache):
    """
    DBCache (Dual Block Cache) implementation for KsanaDiT.

    This cache implementation uses the FnBn architecture:
    - _f_n blocks: First n blocks always compute for stable L1 diff calculation
    - _m_n blocks: Middle blocks may be skipped if L1 diff < threshold
    - _b_n blocks: Last n blocks always compute for precision refinement
    """

    def __init__(
        self,
        model_key: KsanaModelKey,
        config: DBCacheConfig,
    ):
        super().__init__(model_key.name, config)
        config = evolve_with_recommend(config, RECOMMEND_DBCACHE_CONFIGS[model_key])
        self.config = config
        self.num_blocks = config.num_blocks
        self.context = DBCacheContext(config, config.num_blocks)

        assert config.fn_compute_blocks >= 0, "fn_compute_blocks must be >= 0"
        assert config.bn_compute_blocks >= 0, "bn_compute_blocks must be >= 0"
        assert config.fn_compute_blocks + config.bn_compute_blocks <= config.num_blocks, (
            f"_f_n({config.fn_compute_blocks}) + _b_n({config.bn_compute_blocks}) must "
            + f"be <= num_blocks({config.num_blocks})"
        )

        log.info(f"DBCache initialized: {config}")

    def reset(self):
        self.context.reset()

    @disable_dynamo()
    def valid_for(self, phase: str, **kwargs) -> bool:
        self.context.mark_step_begin(phase)

        if self.context.is_warmup_step():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: warmup phase, force compute")
            return False

        buffer = self.context.buffers[phase]
        if buffer.f_n_residual is None:
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: no previous cache, force compute")
            return False

        if self.context.should_force_compute():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: continuous cache limit reached")
            self.context.continuous_cached_steps = 0
            return False

        if self.context.has_exceeded_max_cached_steps():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: max cached steps reached")
            return False

        return True

    @disable_dynamo()
    def compute_diff_and_decide(
        self,
        phase: str,
        current_f_n_residual: torch.Tensor,
        parallelized: bool = False,
    ) -> bool:
        buffer = self.context.buffers[phase]

        if buffer.f_n_residual is None:
            return False

        prev_residual = buffer.f_n_residual

        if prev_residual.shape != current_f_n_residual.shape:
            log.debug(f"[DBCache] shape mismatch: prev={prev_residual.shape}, curr={current_f_n_residual.shape}")
            return False

        diff = torch.abs(current_f_n_residual - prev_residual)

        current_magnitude = torch.abs(current_f_n_residual).mean() + 1e-8
        prev_magnitude = torch.abs(prev_residual).mean() + 1e-8
        avg_magnitude = (current_magnitude + prev_magnitude) / 2
        relative_diff = diff.mean() / avg_magnitude

        if parallelized:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(relative_diff, op=dist.ReduceOp.AVG)

        diff_value = relative_diff.item()
        can_cache = diff_value < self.config.residual_diff_threshold

        if not hasattr(self, "_diff_history"):
            self._diff_history = []
        self._diff_history.append(diff_value)

        log.info(
            f"[DBCache] step={self.context.current_step} phase={phase}: "
            f"diff={diff_value:.4f}, threshold={self.config.residual_diff_threshold}, "
            f"cache={'HIT' if can_cache else 'MISS'}"
        )

        return can_cache

    @disable_dynamo()
    def try_get_prev_cache(self, phase: str, **kwargs):
        buffer = self.context.buffers[phase]

        if buffer.b_n_residual is None:
            return None, None

        self.context.cached_steps[phase] += 1
        self.context.cached_steps_count += 1
        self.context.continuous_cached_steps += 1

        if self.config.enable_taylorseer and len(buffer.prev_bn_residuals) > 0:
            calibrated_residual = self._apply_taylorseer(buffer)
            return calibrated_residual, buffer.b_n_encoder_residual

        return buffer.b_n_residual, buffer.b_n_encoder_residual

    def _apply_taylorseer(self, buffer: CacheBuffer) -> torch.Tensor:
        order = min(self.config.taylorseer_order, len(buffer.prev_bn_residuals))
        if order == 0:
            return buffer.b_n_residual

        result = buffer.b_n_residual.clone()

        if order >= 1 and len(buffer.prev_bn_residuals) >= 1:
            delta = buffer.b_n_residual - buffer.prev_bn_residuals[-1]
            result = result + delta

        return result

    @disable_dynamo()
    def __call__(
        self,
        phase: str,
        x: torch.Tensor,
        step_iter: int,
        timestep: int,
        blocks: list,
        **kwargs,
    ) -> torch.Tensor:
        if blocks is None:
            return x
        use_cache = False
        step = self.context.current_step
        f_n_start, f_n_end = self.context.fn_blocks_range
        m_n_start, m_n_end = self.context.mn_blocks_range
        b_n_start, b_n_end = self.context.bn_blocks_range

        if self.valid_for(phase, x=x, step_iter=step_iter, timestep=timestep):
            x_ori = x.clone()
            for i in range(f_n_start, min(f_n_end, len(blocks))):
                x = blocks[i](x, **kwargs)

            f_n_residual = x - x_ori

            if self.compute_diff_and_decide(phase, f_n_residual):
                b_n_residual, _ = self.try_get_prev_cache(phase, x=x, step_iter=step_iter, timestep=timestep)
                base_info = f"step={step} phase={phase}"
                if b_n_residual is not None:
                    x = x + b_n_residual.to(x.device)
                    use_cache = True
                    log.info(f"{base_info} cache=HIT skip _m_n[{m_n_start},{m_n_end}) _b_n[{b_n_start},{b_n_end})")
                else:
                    log.info(f"{base_info} cache=MISS " f"reason=no_cached_residual")

            if not use_cache:
                x_before_m_n = x.clone()

                for i in range(m_n_start, min(m_n_end, len(blocks))):
                    x = blocks[i](x, **kwargs)

                for i in range(b_n_start, min(b_n_end, len(blocks))):
                    x = blocks[i](x, **kwargs)

                b_n_residual = x - x_before_m_n
                self.update_states(phase, timestep, f_n_residual, b_n_residual)
                log.info(
                    f"[DBCache] step={step} phase={phase} cache=MISS "
                    f"computed _m_n[{m_n_start},{m_n_end}) _b_n[{b_n_start},{b_n_end})"
                )
        else:
            x_ori = x.clone()
            for block in blocks:
                x = block(x, **kwargs)
            full_residual = x - x_ori
            self.update_states(phase, timestep, full_residual, full_residual)
            log.info(f"[DBCache] step={step} phase={phase} cache=MISS " f"warmup compute all blocks ({len(blocks)})")
        self.advance_step_once(timestep)
        return x

    @disable_dynamo()
    def update_states(
        self,
        phase: str,
        current_timestep: int,
        f_n_residual: torch.Tensor,
        b_n_residual: torch.Tensor,
        b_n_encoder_residual: torch.Tensor | None = None,
    ):
        buffer = self.context.buffers[phase]

        if self.config.enable_taylorseer:
            if buffer.b_n_residual is not None:
                buffer.prev_bn_residuals.append(buffer.b_n_residual.clone())
                max_history = self.config.taylorseer_order + 1
                if len(buffer.prev_bn_residuals) > max_history:
                    buffer.prev_bn_residuals.pop(0)

        buffer.f_n_residual = f_n_residual.clone()
        buffer.b_n_residual = b_n_residual.clone()
        if b_n_encoder_residual is not None:
            buffer.b_n_encoder_residual = b_n_encoder_residual.clone()

        self.context.continuous_cached_steps = 0

    def advance_step(self):
        self.context.current_step += 1

    def advance_step_once(self, timestep: int | float | torch.Tensor):
        try:
            ts_val = int(timestep) if isinstance(timestep, (int, float)) else int(timestep.item())
        except (ValueError, TypeError, RuntimeError) as e:
            log.warning("Failed to convert timestep %s: %s", timestep, e)
            ts_val = None

        if ts_val is not None:
            if self.context.last_advanced_timestep == ts_val:
                return
            self.context.last_advanced_timestep = ts_val

        self.advance_step()

    @disable_dynamo()
    def show_cache_rate(self):
        cond_total = self.context.total_steps["cond"]
        cond_cached = self.context.cached_steps["cond"]
        uncond_total = self.context.total_steps["uncond"]
        uncond_cached = self.context.cached_steps["uncond"]

        cond_rate = 100 * cond_cached / max(cond_total, 1)
        uncond_rate = 100 * uncond_cached / max(uncond_total, 1)

        # Compute diff statistics
        diff_stats = ""
        if hasattr(self, "_diff_history") and len(self._diff_history) > 0:
            import numpy as np

            diffs = np.array(self._diff_history)
            diff_stats = (
                f"\n[DBCache] Diff Stats: min={diffs.min():.4f}, max={diffs.max():.4f}, "
                f"mean={diffs.mean():.4f}, median={np.median(diffs):.4f}, "
                f"p25={np.percentile(diffs, 25):.4f}, p75={np.percentile(diffs, 75):.4f}"
            )
            # Suggest optimal threshold
            suggested_threshold = np.percentile(diffs, 50)  # median
            diff_stats += f"\n[DBCache] Suggested threshold (median): {suggested_threshold:.4f}"

        log.info(
            f"[DBCache] {self.model_key} Stats: "
            f"_f_n={self.config.fn_compute_blocks}, _b_n={self.config.bn_compute_blocks}, "
            f"threshold={self.config.residual_diff_threshold}, "
            f"cond(cached={cond_cached}/{cond_total}, rate={cond_rate:.1f}%), "
            f"uncond(cached={uncond_cached}/{uncond_total}, rate={uncond_rate:.1f}%)"
            f"{diff_stats}"
        )

    def get_stats_summary(self) -> dict:
        cond_total = self.context.total_steps["cond"]
        cond_cached = self.context.cached_steps["cond"]
        uncond_total = self.context.total_steps["uncond"]
        uncond_cached = self.context.cached_steps["uncond"]

        return {
            "name": self.model_key,
            "config": str(self.config),
            "cond_total_steps": cond_total,
            "cond_cached_steps": cond_cached,
            "cond_cache_rate": cond_cached / max(cond_total, 1),
            "uncond_total_steps": uncond_total,
            "uncond_cached_steps": uncond_cached,
            "uncond_cache_rate": uncond_cached / max(uncond_total, 1),
        }

    def offload_to_cpu(self):
        for buffer in self.context.buffers.values():
            if buffer.f_n_residual is not None:
                buffer.f_n_residual = buffer.f_n_residual.cpu()
            if buffer.b_n_residual is not None:
                buffer.b_n_residual = buffer.b_n_residual.cpu()
            if buffer.b_n_encoder_residual is not None:
                buffer.b_n_encoder_residual = buffer.b_n_encoder_residual.cpu()
            for i, t in enumerate(buffer.prev_fn_residuals):
                buffer.prev_fn_residuals[i] = t.cpu()
            for i, t in enumerate(buffer.prev_bn_residuals):
                buffer.prev_bn_residuals[i] = t.cpu()
