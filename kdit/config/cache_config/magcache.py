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

from .base import KsanaStepCacheConfig


@dataclass
class MagCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="MagCache")
    threshold: float | None = field(default=None)
    k: int | None = field(default=None)
    max_skip_steps: int | None = field(default=None)
    retention_ratio: float | None = field(default=None)
    mode: str | None = field(default=None)
    split_step: int | None = field(default=None)
    mag_ratios: list | None = field(default=None)
    cache_device: str | None = field(default=None)
    start_step: int | None = field(default=None)
    end_step: int | None = field(default=None)
    verbose: bool | None = field(default=None)
