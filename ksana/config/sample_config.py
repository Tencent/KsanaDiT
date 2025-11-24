from dataclasses import dataclass, field


@dataclass(frozen=True)
class KsanaSampleConfig:
    steps: int | None = field(default=None)
    cfg_scale: float | tuple[float, float] | None = field(default=None)
    shift: float | None = field(default=None)
    solver: str | None = field(default=None)
    denoise: float | None = field(default=1.0)
