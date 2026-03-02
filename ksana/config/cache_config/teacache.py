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
class TeaCacheConfig(KsanaStepCacheConfig):
    name: str = field(default="TeaCache")
    threshold: float = field(default=0.2)
    mode: str = field(default="t2v")
    start_step: int = field(default=0)
    end_step: int | None = field(default=None)
    cache_device: str | None = field(default=None)
    verbose: bool = field(default=False)
