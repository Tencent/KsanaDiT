from dataclasses import dataclass, field
from easydict import EasyDict
from .model_config import KsanaModelConfig


@dataclass()
class KsanaPipelineConfig:
    model_name: str = field(default=None)  # wan2.2
    task_type: str = field(default=None)  # t2v, t2i, v2v
    model_size: str = field(default=None)  # A14B, 5B

    default_config: dict | EasyDict | None = field(default=None)
    model_config: KsanaModelConfig = field(default=None)
