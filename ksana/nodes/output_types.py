from dataclasses import dataclass, field

import torch

from ksana.models.model_key import KsanaModelKey


@dataclass
class KsanaNodeModelLoaderOutput:
    model: KsanaModelKey | list[KsanaModelKey] = field(default=None)
    model_name: str = field(default_factory=str)  # TODO(qian):  need remove
    run_dtype: torch.dtype | None = field(default=None)


@dataclass
class KsanaNodeGeneratorOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)


@dataclass
class KsanaNodeVAEEncodeOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)
    batch_size_per_prompt: int = field(default=1)
