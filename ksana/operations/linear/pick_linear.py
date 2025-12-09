from ksana.utils import supports_fp8_compute, log
from .linear import CUBLAS_IS_AVAILABLE, Linear
from .fp8_linear import Fp8Linear, scaled_fp8_ops
import logging
import torch


# Try to find fp8 dtype from state_dict
def find_fp8_dtype(state_dict):
    for _, value in state_dict.items():
        if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return value.dtype
    log.warning("Could not find fp8 dtype in state_dict")
    return None


# TODO(rockcao): 统一fp8_linear和scaled_fp8_ops的逻辑
def pick_linear(run_dtype, state_dict, linear_backend: str, load_device=None):
    support_fp8_compute = supports_fp8_compute(load_device)
    fp8_gemm, scaled_fp8 = False, None
    if linear_backend == "fp8_gemm":
        fp8_gemm = True
        scaled_fp8 = find_fp8_dtype(state_dict)
        if scaled_fp8 is None:
            raise ValueError("Could not find fp8 dtype in state_dict when linear_backend is fp8_gemm")

    if scaled_fp8 is not None:
        log.info(f"Using scaled fp8 ops with dtype {scaled_fp8}")
        return scaled_fp8_ops(
            fp8_matrix_mult=support_fp8_compute and fp8_gemm, scale_input=fp8_gemm, override_dtype=scaled_fp8
        )

    if support_fp8_compute and fp8_gemm:
        return Fp8Linear

    if CUBLAS_IS_AVAILABLE and run_dtype == torch.float16:
        from .linear import CublassLinear

        logging.info("Using cublas ops")
        return CublassLinear

    return Linear
