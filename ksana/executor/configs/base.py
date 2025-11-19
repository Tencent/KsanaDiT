from dataclasses import dataclass, field
from easydict import EasyDict


@dataclass
class KsanaExecutorConfig:
    """
    Base configuration class for Ksana executors.
    """

    default_model_config: dict | EasyDict = field(default=None)

    steps: int = field(default=50)
    cfg_scale: float = field(default=None)
    sample_shift: float = field(default=None)
    sample_solver: str | None = field(default=None)  # TODO: change to class
