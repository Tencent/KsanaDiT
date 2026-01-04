from dataclasses import dataclass, field

import torch

from .torch_compile_config import KsanaTorchCompileConfig
from ..operations import KsanaAttentionBackend


@dataclass()
class KsanaModelConfig:
    run_dtype: torch.dtype | str = field(default=torch.float16)
    linear_backend: str | None = field(default="fp16_gemm")
    attn_backend: KsanaAttentionBackend | None = field(default=KsanaAttentionBackend.FLASH_ATTN)
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
        assert self.linear_backend in [
            "default",
            "fp8_gemm",
            "fp8_gemm_dynamic",
            "fp16_gemm",
        ], f"linear_backend {self.linear_backend} not supported"
        if not KsanaAttentionBackend.support(self.attn_backend):
            raise ValueError(
                f"attn_backend {self.attn_backend} not supported in {KsanaAttentionBackend.get_supported_list()}"
            )
