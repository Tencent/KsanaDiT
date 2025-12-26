from dataclasses import dataclass, field
from .base import KsanaStepCacheConfig


@dataclass
class EasyCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="EasyCache")
    reuse_thresh: float | None = field(default=None)
    start_percent: float | None = field(default=None)
    end_percent: float | None = field(default=None)
    verbose: bool | None = field(default=None)
