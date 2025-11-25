from dataclasses import dataclass, field

# from easydict import EasyDict
import torch

from .torch_compile_config import KsanaTorchCompileConfig


@dataclass()
class KsanaModelConfig:
    weight_dtype: torch.dtype | str = field(default="default")
    linear_backend: str | None = field(default="fp16_gemm")
    attn_backend: str | None = field(default="flash_attention")
    torch_compile_config: KsanaTorchCompileConfig | None = field(default=None)

    def __post_init__(self):
        assert self.weight_dtype in [
            torch.float16,
            torch.bfloat16,
            "float16",
            "bfloat16",
            "default",
        ], f"weight_dtype {self.weight_dtype} not supported"
        assert self.linear_backend in [
            "default",
            "fp8_gemm",
            "fp16_gemm",
        ], f"linear_backend {self.linear_backend} not supported"
        assert self.attn_backend in [
            "flash_attention",
            "sage_attention",
        ], f"attn_backend {self.attn_backend} not supported"
