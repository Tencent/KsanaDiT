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

from .base import KsanaBlockCacheConfig


@dataclass
class DBCacheConfig(KsanaBlockCacheConfig):
    name: str = field(default="DBCache")
    fn_compute_blocks: int | None = field(default=None)
    bn_compute_blocks: int | None = field(default=None)
    residual_diff_threshold: float | None = field(default=None)
    max_warmup_steps: int | None = field(default=None)
    warmup_interval: int | None = field(default=None)
    max_cached_steps: int | None = field(default=None)
    max_continuous_cached_steps: int | None = field(default=None)
    enable_separate_cfg: bool = field(default=True)
    cfg_compute_first: bool = field(default=False)
    enable_taylorseer: bool = field(default=False)
    taylorseer_order: int = field(default=1)
    num_blocks: int | None = field(default=None)
