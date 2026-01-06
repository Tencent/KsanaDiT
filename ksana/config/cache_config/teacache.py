from dataclasses import dataclass, field

from .base import KsanaStepCacheConfig


@dataclass
class TeaCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="TeaCache")
    rel_l1_thresh: float | None = field(default=None)
    cache_device: str | None = field(default=None)
    start_step: int | None = field(default=None)
    end_step: int | None = field(default=None)
    use_coeffecients: bool | None = field(default=None)
    mode: str | None = field(default=None)
