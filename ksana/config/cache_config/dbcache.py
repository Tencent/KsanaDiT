from dataclasses import dataclass, field
from .base import KsanaBlockCacheConfig


@dataclass
class DBCacheConfig(KsanaBlockCacheConfig):
    name: str = field(default="DBCache")
    Fn_compute_blocks: int | None = field(default=None)
    Bn_compute_blocks: int | None = field(default=None)
    residual_diff_threshold: float | None = field(default=None)
    max_warmup_steps: int | None = field(default=None)
    warmup_interval: int | None = field(default=None)
    max_cached_steps: int | None = field(default=None)
    max_continuous_cached_steps: int | None = field(default=None)
    enable_separate_cfg: bool = field(default=True)
    cfg_compute_first: bool = field(default=False)
    enable_taylorseer: bool = field(default=False)
    taylorseer_order: int = field(default=1)
    num_blocks: int | None = field(default=None)
