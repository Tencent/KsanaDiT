# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch

from ...config import KsanaLinearBackend
from ...utils import log, supports_fp8_compute
from .fp8_linear import Fp8Linear, scaled_fp8_ops
from .linear import CUBLAS_IS_AVAILABLE, Linear


# Try to find fp8 info from state_dict
def find_fp8_info(state_dict):
    fp8_dtype = None
    is_scaled_fp8 = False
    for _, value in state_dict.items():
        if value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            fp8_dtype = value.dtype
            break
    for key, _ in state_dict.items():
        if key.endswith("scale_weight"):
            is_scaled_fp8 = True
            break

    if "scaled_fp8" in state_dict:
        state_dict.pop("scaled_fp8")
        if not is_scaled_fp8:
            log.warning("scaled_fp8 is in state_dict, but no scale_weight found in state_dict")

    log.info(f"find_fp8_info fp8_dtype: {fp8_dtype}, is_scaled_fp8: {is_scaled_fp8}")
    return fp8_dtype, is_scaled_fp8


# TODO(rockcao): 统一fp8_linear和scaled_fp8_ops的逻辑
def pick_linear(run_dtype, state_dict, linear_backend: str, load_device=None):
    support_fp8_compute = supports_fp8_compute(load_device)
    fp8_dtype, is_scaled_fp8 = find_fp8_info(state_dict)

    fp8_gemm = False
    if linear_backend == KsanaLinearBackend.DEFAULT:
        fp8_gemm = fp8_dtype is not None

    if linear_backend == KsanaLinearBackend.FP8_GEMM:
        fp8_gemm = True
        if fp8_dtype is None:
            raise ValueError("Could not find fp8 dtype in state_dict when linear_backend is fp8_gemm")

    if is_scaled_fp8 and fp8_gemm:
        log.info(f"Using scaled fp8 ops with dtype {fp8_dtype}")
        return scaled_fp8_ops(
            fp8_matrix_mult=support_fp8_compute and fp8_gemm, scale_input=fp8_gemm, override_dtype=fp8_dtype
        )

    if support_fp8_compute and fp8_gemm:
        return Fp8Linear

    if CUBLAS_IS_AVAILABLE and run_dtype == torch.float16:
        from .linear import CublassLinear

        logging.info("Using cublas ops")
        return CublassLinear

    return Linear
