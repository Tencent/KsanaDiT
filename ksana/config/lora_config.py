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

from ..utils.const import DEFAULT_LORA_STRENGTH


@dataclass(frozen=True)
class KsanaLoraConfig:
    path: str | None = field(default=None)
    strength: float = field(default=DEFAULT_LORA_STRENGTH)

    def __post_init__(self):
        if self.path is None:
            raise ValueError("path must be specified")
        if self.strength is None or self.strength < 0:
            raise ValueError(f"strength must be a float gt 0, got {self.strength}")
