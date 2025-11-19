from dataclasses import dataclass, field
from .base import KsanaExecutorConfig


@dataclass
class WanExecutorConfig(KsanaExecutorConfig):
    """
    Configuration class for WAN executors.
    """

    steps: int = field(default=50)
    sample_solver: str = field(default="uni_pc")

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


@dataclass
class WanLightLoraExecutorConfig(WanExecutorConfig):
    """
    Configuration class for WAN Lightning executors.
    """

    steps: int = field(default=4)
    cfg_scale: float = field(default=1.0)
    sample_shift: float = field(default=5.0)
    sample_solver: str = field(default="euler")

    # lora
    low_noise_lora_checkpoint: str = field(default="low_noise_model.safetensors")
    high_noise_lora_checkpoint: str = field(default="high_noise_model.safetensors")
