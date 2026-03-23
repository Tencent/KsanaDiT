# Copyright 2026 Tencent
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

from dataclasses import dataclass

from .model_key import ModelKey


@dataclass(frozen=True)
class ModelPoolKey:
    """ModelPool 中的唯一 key = node_id + ModelKey 枚举。

    frozen dataclass 自动生成 __hash__ 和 __eq__，可直接用作 dict key。
    """

    node_id: int
    pin: ModelKey  # 直接用枚举，不用 .value

    @property
    def uid(self) -> str:
        return f"model:{self.node_id}:{self.pin.name}"
