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

from enum import Enum


class TensorKey(str, Enum):
    """tensor_pool 中所有合法的 key。

    继承 str 使得枚举值可以直接用作 dict key / 字符串比较。
    新增 tensor 通道时必须在此注册。
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    IMAGE_EMBEDS = "image_embeds"
    LATENTS = "latents"
    VIDEO = "video"

    # Pipeline → Node 的输入 tensor（通过 engine.put_tensors 写入）
    IMAGE = "image"
    START_IMG = "start_img"
    END_IMG = "end_img"
    INPUT_LATENT = "input_latent"
