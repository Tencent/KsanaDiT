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
class CustomStepCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="CustomStepCache")
    steps: list[int] | None = field(default=None)
    scales: list[float] | None = field(default=None)

    def __post_init__(self):
        if self.steps is None:
            raise ValueError("steps must be provided as list[int]")
        if not isinstance(self.steps, list):
            self.steps = [self.steps]
        if self.scales is None:
            self.scales = 1.0
        if not isinstance(self.scales, list):
            self.scales = [self.scales] * len(self.steps)
