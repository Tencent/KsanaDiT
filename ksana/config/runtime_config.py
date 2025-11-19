from dataclasses import dataclass, field
import torch


@dataclass
class KsanaRuntimeConfig:
    input_size: tuple[int, int] | None = field(default=None)
    input_frame_num: int | None = field(default=None)
    seed: int | None = field(default=None)
    run_dtype: torch.dtype | None = field(default=None)
    offload_model: bool | None = field(default=None)
    # use boundary in wan high and low noise model
    boundary: float | None = field(default=None)
