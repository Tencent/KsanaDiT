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

from dataclasses import dataclass, field

import torch

from .attention_config import KsanaAttentionConfig
from .linear_config import KsanaLinearBackend
from .torch_compile_config import KsanaTorchCompileConfig


@dataclass()
class KsanaModelConfig:
    run_dtype: torch.dtype | str = field(default=torch.float16)
    linear_backend: KsanaLinearBackend = field(default=KsanaLinearBackend.FP16_GEMM)
    attention_config: KsanaAttentionConfig = field(default=KsanaAttentionConfig())
    torch_compile_config: KsanaTorchCompileConfig | None = field(default=None)
    rms_dtype: torch.dtype | str = field(default="float")
    boundary: float | None = field(default=None, metadata={"help": "boundary for if have two models"})

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

        assert self.rms_dtype in [
            "float",
            "half",
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ], f"rms_dtype {self.rms_dtype} not supported"
        if self.rms_dtype == "float":
            self.rms_dtype = torch.float32
        elif self.rms_dtype == "half":
            self.rms_dtype = torch.float16

        if not KsanaLinearBackend.support(self.linear_backend):
            raise ValueError(
                f"linear_backend {self.linear_backend} not supported in {KsanaLinearBackend.get_supported_list()}"
            )
