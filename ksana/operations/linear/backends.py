from __future__ import annotations

from enum import Enum


class KsanaLinearBackend(Enum):
    DEFAULT = "default"
    FP8_GEMM = "fp8_gemm"
    FP8_GEMM_DYNAMIC = "fp8_gemm_dynamic"
    FP16_GEMM = "fp16_gemm"

    @staticmethod
    def get_supported_list() -> list[str]:
        return [b.value for b in KsanaLinearBackend]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaLinearBackend.get_supported_list()
        elif isinstance(type, KsanaLinearBackend):
            return True
        else:
            return False
