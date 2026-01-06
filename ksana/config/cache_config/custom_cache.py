from dataclasses import dataclass, field

from .base import KsanaStepCacheConfig


@dataclass
class CustomStepCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="CustomStepCache")
    steps: list[int] | None = field(default=None)
    scales: list[float] | None = field(default=None)

    def __post_init__(self):
        if self.steps is None:
            raise ValueError("steps must be provided as list[int]")
        if not isinstance(self.steps, list):
            self.steps = [self.steps]
        if self.scales is None:
            self.scales = 1.0
        if isinstance(self.scales, float):
            self.scales = [self.scales] * len(self.steps)
