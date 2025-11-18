from dataclasses import dataclass
from .base import KsanaExecutorConfig


@dataclass
class WanExecutorConfig(KsanaExecutorConfig):
    """
    Configuration class for WAN executors.
    """

    steps: int = 50
    cfg_scale: float = 7.5

    sample_solver: str = "uni_pc"
    default_model_config: None

    def __post_init__(self):
        if self.default_model_config is None:
            return
        assert (
            self.default_model_config.model_name == "wan2.2"
        ), f"model_name {self.default_model_config.model_name} is not supported"
        assert self.default_model_config.task_type in [
            "t2v",
            "t2i",
            "v2v",
        ], f"task_type {self.default_model_config.task_type} is not supported"
        assert self.default_model_config.model_size in [
            "A14B",
            "5B",
        ], f"model_size {self.default_model_config.model_size} is not supported"
