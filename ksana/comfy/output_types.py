import torch
from dataclasses import dataclass, field

from ksana.models.model_key import KsanaModelKey


@dataclass
class KsanaComfyModelLoaderOutput:
    model: KsanaModelKey | list[KsanaModelKey] = field(default=None)
    model_name: str = field(default_factory=str)  # TODO(qian):  need remove
    run_dtype: torch.dtype | None = field(default=None)
    boundary: float | None = field(default=None)


@dataclass
class KsanaComfyGeneratorOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)


@dataclass
class KsanaComfyVAEEncodeOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)
    batch_per_prompt: int = field(default=1)
