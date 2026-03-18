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

import os
from datetime import datetime

from ..generate_inputs import PipelineGenerateInputs


def compute_save_path(
    inputs: PipelineGenerateInputs,
    *,
    prefix: str,
    ext: str,
) -> str | None:
    """从 runtime_config 计算保存路径。

    如果 save_output=False，返回 None（SaveNode 会跳过保存）。

    Args:
        inputs: Pipeline 生成输入。
        prefix: 文件名前缀，如 ``"qwen"`` 或 ``"wan"``。
        ext: 文件扩展名（含点号），如 ``".png"`` 或 ``".mp4"``。
    """
    rc = inputs.runtime_config
    if not rc.save_output:
        return None

    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_text = inputs.prompt if isinstance(inputs.prompt, str) else inputs.prompt[0]
    formatted_prompt = prompt_text.replace(" ", "_").replace("/", "_")[:30]
    out_size = rc.size
    filename = f"{prefix}_w{out_size[0]}_h{out_size[1]}" f"_{formatted_time}_{formatted_prompt}_0{ext}"
    return os.path.join(rc.output_folder, filename)
