"""
DBCache (Dual Block Cache) for KsanaDiT.

This is a deep integration of cache-dit's DBCache algorithm into KsanaDiT,

DBCache leverages the feature similarity between adjacent timesteps in the
diffusion process. By computing L1 diff of residuals after the first Fn blocks,
it decides whether to reuse cached residuals from previous steps or compute
all blocks.

Architecture:
    ┌──────────────┬─────────────────────────┬───────────────────┐
    │   Fn Blocks  │      Mn Blocks          │    Bn Blocks      │
    │   (前n个)    │   (中间blocks)          │    (后n个)        │
    │   总是计算    │   可能被跳过/缓存        │    总是计算        │
    │   计算稳定diff│   复用cached residual   │    修正近似误差    │
    └──────────────┴─────────────────────────┴───────────────────┘

Reference:
    https://github.com/vipshop/cache-dit
"""

from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

import torch

from .base_cache import KsanaCache
from .cache_config import DBCacheConfig
from ..utils import log

__all__ = ["DBCache", "DBCacheContext"]


try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:

    def _dynamo_disable(fn=None):
        return fn if fn is not None else (lambda f: f)


class DBCacheContext:
    """
    Context for managing DBCache state during inference.
    """

    def __init__(self, config: DBCacheConfig, num_blocks: int):
        self.config = config
        self.num_blocks = num_blocks

        # Step counters
        self.current_step = 0
        self.cached_steps_count = 0
        self.continuous_cached_steps = 0
        self.last_advanced_timestep: Optional[int] = None

        # Cache buffers for cond and uncond (CFG)
        self.buffers: Dict[str, CacheBuffer] = {
            "cond": CacheBuffer(),
            "uncond": CacheBuffer(),
        }

        # Statistics
        self.total_steps = {"cond": 0, "uncond": 0}
        self.cached_steps = {"cond": 0, "uncond": 0}

    def reset(self):
        """Reset context for a new inference run."""
        self.current_step = 0
        self.cached_steps_count = 0
        self.continuous_cached_steps = 0
        for buffer in self.buffers.values():
            buffer.clear()
        self.total_steps = {"cond": 0, "uncond": 0}
        self.cached_steps = {"cond": 0, "uncond": 0}

    def mark_step_begin(self, phase: str = "cond"):
        """Mark the beginning of a new step."""
        self.total_steps[phase] += 1

    def is_warmup_step(self) -> bool:
        """Check if current step is in warmup phase."""
        if self.current_step < self.config.max_warmup_steps:
            if self.config.warmup_interval > 1:
                return (self.current_step % self.config.warmup_interval) == 0
            return True
        return False

    def should_force_compute(self) -> bool:
        """Check if we should force compute due to continuous cache limit."""
        if self.config.max_continuous_cached_steps > 0:
            return self.continuous_cached_steps >= self.config.max_continuous_cached_steps
        return False

    def has_exceeded_max_cached_steps(self) -> bool:
        """Check if we've exceeded the maximum cached steps."""
        if self.config.max_cached_steps > 0:
            return self.cached_steps_count >= self.config.max_cached_steps
        return False

    @property
    def Fn_blocks_range(self) -> Tuple[int, int]:
        """Return the range of Fn blocks [start, end)."""
        return (0, self.config.Fn_compute_blocks)

    @property
    def Mn_blocks_range(self) -> Tuple[int, int]:
        """Return the range of Mn blocks [start, end)."""
        Fn = self.config.Fn_compute_blocks
        Bn = self.config.Bn_compute_blocks
        if Bn == 0:
            return (Fn, self.num_blocks)
        return (Fn, self.num_blocks - Bn)

    @property
    def Bn_blocks_range(self) -> Tuple[int, int]:
        """Return the range of Bn blocks [start, end)."""
        Bn = self.config.Bn_compute_blocks
        if Bn == 0:
            return (self.num_blocks, self.num_blocks)
        return (self.num_blocks - Bn, self.num_blocks)


@dataclass
class CacheBuffer:
    """Buffer for storing cached tensors."""

    Fn_residual: Optional[torch.Tensor] = None
    Bn_residual: Optional[torch.Tensor] = None
    Bn_encoder_residual: Optional[torch.Tensor] = None

    # For TaylorSeer calibration
    prev_Fn_residuals: list = field(default_factory=list)
    prev_Bn_residuals: list = field(default_factory=list)

    def clear(self):
        self.Fn_residual = None
        self.Bn_residual = None
        self.Bn_encoder_residual = None
        self.prev_Fn_residuals.clear()
        self.prev_Bn_residuals.clear()


class DBCache(KsanaCache):
    """
    DBCache (Dual Block Cache) implementation for KsanaDiT.

    This cache implementation uses the FnBn architecture:
    - Fn blocks: First n blocks always compute for stable L1 diff calculation
    - Mn blocks: Middle blocks may be skipped if L1 diff < threshold
    - Bn blocks: Last n blocks always compute for precision refinement
    """

    def __init__(
        self,
        model_name: str,
        model_type: str,
        model_size: str,
        config: DBCacheConfig,
    ):
        super().__init__(f"{model_name}_{model_type}_{model_size}_{config.name}")
        self.config = config
        self.num_blocks = config.num_blocks
        self.context = DBCacheContext(config, config.num_blocks)

        # Validate config
        assert config.Fn_compute_blocks >= 0, "Fn_compute_blocks must be >= 0"
        assert config.Bn_compute_blocks >= 0, "Bn_compute_blocks must be >= 0"
        assert (
            config.Fn_compute_blocks + config.Bn_compute_blocks <= config.num_blocks
        ), f"Fn({config.Fn_compute_blocks}) + Bn({config.Bn_compute_blocks}) must be <= num_blocks({config.num_blocks})"

        log.info(f"DBCache initialized: {config}")

    def reset(self):
        """Reset cache for a new inference run."""
        self.context.reset()

    @_dynamo_disable()
    def can_use_cache(self, phase: str, current_x_input: torch.Tensor, current_timestep: int) -> bool:
        """
        Determine if we can use cache for current step.

        This is called BEFORE Fn blocks computation to do early checking.
        The actual L1 diff check is done in compute_diff_and_decide().
        """
        self.context.mark_step_begin(phase)

        # Check warmup phase
        if self.context.is_warmup_step():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: warmup phase, force compute")
            return False

        # Check if previous buffer exists
        buffer = self.context.buffers[phase]
        if buffer.Fn_residual is None:
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: no previous cache, force compute")
            return False

        # Check continuous cache limit
        if self.context.should_force_compute():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: continuous cache limit reached")
            self.context.continuous_cached_steps = 0
            return False

        # Check max cached steps
        if self.context.has_exceeded_max_cached_steps():
            log.debug(f"[DBCache] phase={phase} step={self.context.current_step}: max cached steps reached")
            return False

        return True

    @_dynamo_disable()
    def compute_diff_and_decide(
        self,
        phase: str,
        current_Fn_residual: torch.Tensor,
        parallelized: bool = False,
    ) -> bool:
        """
        Compute L1 diff and decide whether to use cache.

        Args:
            phase: "cond" or "uncond"
            current_Fn_residual: Residual after Fn blocks
            parallelized: Whether running in distributed mode

        Returns:
            True if should use cache, False if should compute
        """
        buffer = self.context.buffers[phase]

        if buffer.Fn_residual is None:
            return False

        # Compute relative L1 diff
        prev_residual = buffer.Fn_residual

        # Handle shape mismatch (e.g., context parallelism)
        if prev_residual.shape != current_Fn_residual.shape:
            log.debug(f"[DBCache] shape mismatch: prev={prev_residual.shape}, curr={current_Fn_residual.shape}")
            return False

        # Compute L1 diff - use absolute diff normalized by mean of current residual
        # This is more stable than relative diff between consecutive residuals
        diff = torch.abs(current_Fn_residual - prev_residual)

        # Method 2: Relative to current residual magnitude (more stable)
        current_magnitude = torch.abs(current_Fn_residual).mean() + 1e-8
        prev_magnitude = torch.abs(prev_residual).mean() + 1e-8
        avg_magnitude = (current_magnitude + prev_magnitude) / 2
        relative_diff = diff.mean() / avg_magnitude

        # In distributed mode, need to sync diff across ranks
        if parallelized:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.all_reduce(relative_diff, op=dist.ReduceOp.AVG)

        diff_value = relative_diff.item()
        can_cache = diff_value < self.config.residual_diff_threshold

        # Record diff for statistics
        if not hasattr(self, "_diff_history"):
            self._diff_history = []
        self._diff_history.append(diff_value)

        # Always log diff info at INFO level for debugging
        log.info(
            f"[DBCache] step={self.context.current_step} phase={phase}: "
            f"diff={diff_value:.4f}, threshold={self.config.residual_diff_threshold}, "
            f"cache={'HIT' if can_cache else 'MISS'}"
        )

        return can_cache

    @_dynamo_disable()
    def try_get_prev_cache(self, phase: str, current_x_input: torch.Tensor, current_timestep: int):
        """
        Try to get cached residual for current step.

        Returns:
            Tuple of (Bn_residual, Bn_encoder_residual) if cache is available,
            (None, None) otherwise.
        """
        buffer = self.context.buffers[phase]

        if buffer.Bn_residual is None:
            return None, None

        # Update statistics
        self.context.cached_steps[phase] += 1
        self.context.cached_steps_count += 1
        self.context.continuous_cached_steps += 1

        # Apply TaylorSeer calibration if enabled
        if self.config.enable_taylorseer and len(buffer.prev_Bn_residuals) > 0:
            calibrated_residual = self._apply_taylorseer(buffer)
            return calibrated_residual, buffer.Bn_encoder_residual

        return buffer.Bn_residual, buffer.Bn_encoder_residual

    def _apply_taylorseer(self, buffer: CacheBuffer) -> torch.Tensor:
        """Apply TaylorSeer calibration to improve cached residual accuracy."""
        order = min(self.config.taylorseer_order, len(buffer.prev_Bn_residuals))
        if order == 0:
            return buffer.Bn_residual

        # Taylor series approximation
        # F_pred(t-k) ≈ F(t) + Σ(Δ^i F(t) / i!) * (-k)^i
        result = buffer.Bn_residual.clone()

        if order >= 1 and len(buffer.prev_Bn_residuals) >= 1:
            # First order: add the difference
            delta = buffer.Bn_residual - buffer.prev_Bn_residuals[-1]
            result = result + delta

        return result

    @_dynamo_disable()
    def update_states(
        self,
        phase: str,
        current_timestep: int,
        Fn_residual: torch.Tensor,
        Bn_residual: torch.Tensor,
        Bn_encoder_residual: Optional[torch.Tensor] = None,
    ):
        """
        Update cache buffers after computing all blocks.

        Args:
            phase: "cond" or "uncond"
            current_timestep: Current diffusion timestep
            Fn_residual: Residual after Fn blocks (for diff calculation)
            Bn_residual: Residual of Mn blocks (to cache)
            Bn_encoder_residual: Encoder residual (for dual-stream models)
        """
        buffer = self.context.buffers[phase]

        # Store for TaylorSeer
        if self.config.enable_taylorseer:
            if buffer.Bn_residual is not None:
                buffer.prev_Bn_residuals.append(buffer.Bn_residual.clone())
                # Keep only last N residuals
                max_history = self.config.taylorseer_order + 1
                if len(buffer.prev_Bn_residuals) > max_history:
                    buffer.prev_Bn_residuals.pop(0)

        # Update buffers
        buffer.Fn_residual = Fn_residual.clone()
        buffer.Bn_residual = Bn_residual.clone()
        if Bn_encoder_residual is not None:
            buffer.Bn_encoder_residual = Bn_encoder_residual.clone()

        # Reset continuous cache counter
        self.context.continuous_cached_steps = 0

    def advance_step(self):
        """Advance to next diffusion step."""
        self.context.current_step += 1

    def advance_step_once(self, timestep: Any):
        try:
            ts_val = int(timestep) if isinstance(timestep, (int, float)) else int(timestep.item())
        except Exception:
            ts_val = None

        if ts_val is not None:
            if self.context.last_advanced_timestep == ts_val:
                return
            self.context.last_advanced_timestep = ts_val

        self.advance_step()

    @_dynamo_disable()
    def show_cache_rate(self):
        """Print cache statistics."""
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
            f"[DBCache] {self.name} Stats: "
            f"Fn={self.config.Fn_compute_blocks}, Bn={self.config.Bn_compute_blocks}, "
            f"threshold={self.config.residual_diff_threshold}, "
            f"cond(cached={cond_cached}/{cond_total}, rate={cond_rate:.1f}%), "
            f"uncond(cached={uncond_cached}/{uncond_total}, rate={uncond_rate:.1f}%)"
            f"{diff_stats}"
        )

    def get_stats_summary(self) -> Dict[str, Any]:
        """Return cache statistics as a dictionary."""
        cond_total = self.context.total_steps["cond"]
        cond_cached = self.context.cached_steps["cond"]
        uncond_total = self.context.total_steps["uncond"]
        uncond_cached = self.context.cached_steps["uncond"]

        return {
            "name": self.name,
            "config": str(self.config),
            "cond_total_steps": cond_total,
            "cond_cached_steps": cond_cached,
            "cond_cache_rate": cond_cached / max(cond_total, 1),
            "uncond_total_steps": uncond_total,
            "uncond_cached_steps": uncond_cached,
            "uncond_cache_rate": uncond_cached / max(uncond_total, 1),
        }

    def clone_input_x(self, current_timestep: int, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def offload_to_cpu(self):
        for buffer in self.context.buffers.values():
            if buffer.Fn_residual is not None:
                buffer.Fn_residual = buffer.Fn_residual.cpu()
            if buffer.Bn_residual is not None:
                buffer.Bn_residual = buffer.Bn_residual.cpu()
            if buffer.Bn_encoder_residual is not None:
                buffer.Bn_encoder_residual = buffer.Bn_encoder_residual.cpu()
            for i, t in enumerate(buffer.prev_Fn_residuals):
                buffer.prev_Fn_residuals[i] = t.cpu()
            for i, t in enumerate(buffer.prev_Bn_residuals):
                buffer.prev_Bn_residuals[i] = t.cpu()
