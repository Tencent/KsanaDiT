from dataclasses import dataclass


@dataclass
class KsanaExecutorConfig:
    """
    Base configuration class for Ksana executors.
    """

    default_model_config = None

    # model_path: str = None
    sample_solver: str | None = None  # TODO: change to class
    steps: int = 50
    cfg_scale: float = 7.5
