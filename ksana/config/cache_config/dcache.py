from dataclasses import dataclass, field
from .base import KsanaStepCacheConfig


@dataclass
class DCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="DCache")
    fast_degree: int | None = field(default=None)
    slow_degree: int | None = field(default=None)
    fast_force_calc_every_n_step: int | None = field(default=None)
    slow_force_calc_every_n_step: int | None = field(default=None)
    skip_first_n_iter: int = field(default=2)
