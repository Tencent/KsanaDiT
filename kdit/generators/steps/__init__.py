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

"""Generator 积木函数 — 无状态工具函数。

这些函数不依赖 Generator 实例状态，可独立测试和复用。
"""

from .noise import create_cache, create_random_noise_latents
from .tensor_ops import cast_to, split_tensors
from .validation import (
    valid_aux_latent,
    valid_cache_config,
    valid_diffusion_model,
    valid_runtime_config,
    valid_sample_config,
)

__all__ = [
    # validation
    "valid_diffusion_model",
    "valid_sample_config",
    "valid_cache_config",
    "valid_runtime_config",
    "valid_aux_latent",
    # noise
    "create_random_noise_latents",
    "create_cache",
    # tensor_ops
    "split_tensors",
    "cast_to",
]
