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
class DCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="DCache")
    fast_degree: int | None = field(default=None)
    slow_degree: int | None = field(default=None)
    fast_force_calc_every_n_step: int | None = field(default=None)
    slow_force_calc_every_n_step: int | None = field(default=None)
    skip_first_n_iter: int = field(default=2)
