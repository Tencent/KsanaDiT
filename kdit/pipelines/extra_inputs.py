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

"""ExtraInputs — Pipeline 模型特有输入的基类。

每个 Pipeline 定义自己的 ExtraInputs 子类（如 WanI2VExtraInputs）。
T2V/T2I 等无特有输入的 Pipeline 不需要传此参数。

用法::

    pipeline.generate(
        prompts,
        extra_inputs=WanI2VExtraInputs(start_img_path="a.jpg"),
        sample_config=...,
    )
"""

from dataclasses import dataclass


@dataclass
class ExtraInputs:
    """模型特有输入的基类。

    每个 Pipeline 定义自己的子类，包含该 Pipeline 特有的输入字段。
    T2V/T2I 等无特有输入的 Pipeline 不需要传此参数。
    """
