from ksana.utils import supports_fp8_compute
from .linear import CUBLAS_IS_AVAILABLE, Linear
from .fp8_linear import Fp8Linear, scaled_fp8_ops
import logging
import torch


def pick_linear(run_dtype, load_device=None, fp8_gemm=False, scaled_fp8=None):
    support_fp8_compute = supports_fp8_compute(load_device)

    if scaled_fp8 is not None:
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
