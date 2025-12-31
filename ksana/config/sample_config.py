from dataclasses import dataclass, field
from ..sample_solvers import SUPPORTED_SOLVERS
from ..utils.const import DEFAULT_DENOISE, DEFAULT_BATCH_PER_PROMPT


@dataclass(frozen=True)
class KsanaSampleConfig:
    steps: int | None = field(default=None)
    cfg_scale: float | tuple[float, float] | None = field(default=None)
    shift: float | None = field(default=None)
    solver: str | None = field(default=None)
    denoise: float | None = field(default=DEFAULT_DENOISE)
    sigmas: list[float] | None = field(default=None)
    batch_per_prompt: int | list[int] | None = field(default=DEFAULT_BATCH_PER_PROMPT)

    def __post_init__(self):
        assert (
            self.solver in SUPPORTED_SOLVERS or self.solver is None
        ), f"unsupported solver {self.solver}, not in {SUPPORTED_SOLVERS}"
