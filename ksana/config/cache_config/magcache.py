from dataclasses import dataclass, field
from .base import KsanaStepCacheConfig


@dataclass
class MagCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="MagCache")
    threshold: float | None = field(default=None)
    K: int | None = field(default=None)
    cache_device: str | None = field(default=None)
    start_step: int | None = field(default=None)
    end_step: int | None = field(default=None)
    verbose: bool | None = field(default=None)
