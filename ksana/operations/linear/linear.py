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

import torch


class Linear(torch.nn.Linear):
    def reset_parameters(self):
        return None


CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear

    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:

    class CublassLinear(CublasLinear):
        def reset_parameters(self):
            return None

        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)
