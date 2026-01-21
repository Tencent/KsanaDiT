from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..utils.const import DEFAULT_DENOISE


class KsanaSolverType(Enum):
    UNI_PC = "uni_pc"
    EULER = "euler"
    FLOWMATCH_EULER = "flowmatch_euler"
    # DPM_PLUS_PLUS = "dpm++" #TODO: 这个还需要check，里面有两次 shift 操作，是否合理

    @staticmethod
    def get_supported_list() -> list[str]:
        return [b.value for b in KsanaSolverType]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaSolverType.get_supported_list()
        elif isinstance(type, KsanaSolverType):
            return True
        else:
            return False


@dataclass(frozen=True)
class KsanaSampleConfig:
    steps: int | None = field(default=None)
    cfg_scale: float | list[float, float] | None = field(default=None)
    shift: float | None = field(default=None)
    solver: KsanaSolverType | None = field(default=None)
    denoise: float | None = field(default=DEFAULT_DENOISE)
    sigmas: list[float] | None = field(default=None)

    def __post_init__(self):
        if self.solver is not None and not isinstance(self.solver, KsanaSolverType):
            raise ValueError(f"solver must be a KsanaSolverType enum, got {type(self.solver)}")
