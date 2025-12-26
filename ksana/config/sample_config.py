from dataclasses import dataclass, field
from easydict import EasyDict
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

    @staticmethod
    def copy_with_default(input_config, default: dict | EasyDict):
        return KsanaSampleConfig(
            steps=default.get("steps", None) if input_config.steps is None else input_config.steps,
            cfg_scale=(default.get("cfg_scale", None) if input_config.cfg_scale is None else input_config.cfg_scale),
            shift=default.get("sample_shift", None) if input_config.shift is None else input_config.shift,
            solver=default.get("sample_solver", None) if input_config.solver is None else input_config.solver,
            denoise=default.get("denoise", None) if input_config.denoise is None else input_config.denoise,
            sigmas=default.get("sigmas", None) if input_config.sigmas is None else input_config.sigmas,
            batch_per_prompt=input_config.batch_per_prompt,
        )
