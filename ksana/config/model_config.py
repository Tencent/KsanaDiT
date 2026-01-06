from dataclasses import dataclass, field

import torch

from ..operations.linear import KsanaLinearBackend
from .attention_config import KsanaAttentionConfig
from .torch_compile_config import KsanaTorchCompileConfig


@dataclass()
class KsanaModelConfig:
    run_dtype: torch.dtype | str = field(default=torch.float16)
    linear_backend: KsanaLinearBackend = field(default=KsanaLinearBackend.FP16_GEMM)
    attention_config: KsanaAttentionConfig = field(default=KsanaAttentionConfig())
    torch_compile_config: KsanaTorchCompileConfig | None = field(default=None)

    def __post_init__(self):
        assert self.run_dtype in [
            torch.float16,
            torch.bfloat16,
            "float16",
            "bfloat16",
        ], f"run_dtype {self.run_dtype} not supported"
        if self.run_dtype == "float16":
            self.run_dtype = torch.float16
        if self.run_dtype == "bfloat16":
            self.run_dtype = torch.bfloat16
        if not KsanaLinearBackend.support(self.linear_backend):
            raise ValueError(
                f"linear_backend {self.linear_backend} not supported in {KsanaLinearBackend.get_supported_list()}"
            )
