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
