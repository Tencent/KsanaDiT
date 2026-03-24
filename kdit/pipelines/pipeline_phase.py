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

"""Pipeline 声明式定义 — LoadTask."""


from dataclasses import dataclass

from kdit.models.model_key import ModelKey


@dataclass(frozen=True)
class LoadTask:
    """模型加载阶段 — 声明一个需要加载的模型。

    Attributes:
        model_key: 具体的模型 key（ModelKey 枚举值）。
    """

    model_key: ModelKey
