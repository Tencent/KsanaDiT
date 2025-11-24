from dataclasses import dataclass, field
import torch


@dataclass
class KsanaRuntimeConfig:
    """_summary_
    size: tuple[int, int] = field(default=None): target image or video image size
    """

    size: tuple[int, int] | None = field(default=None)
    frame_num: int | None = field(default=None)
    seed: int | None = field(default=None)
    run_dtype: torch.dtype | None = field(default=None)
    offload_model: bool | None = field(default=None)
    # use boundary in wan high and low noise model
    boundary: float | None = field(default=None)

    fsdp: bool = field(default=False, metadata={"help": "use fully sharded data parallel"})
