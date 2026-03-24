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


class TensorKey(Enum):
    """tensor_pool 中所有合法的 key。

    新增 tensor 通道时必须在此注册。
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    LATENTS = "latents"
    VIDEO = "video"

    # Pipeline → Node 的输入 tensor（通过 engine.put_tensors 写入）
    IMAGE = "image"
    START_IMG = "start_img"
    END_IMG = "end_img"

    # 主 latent + 可选 mask（list 形式存储于 tensor_pool）
    BASE_LATENT = "base_latent"
    # 辅助 latent 输入（Qwen img_emb / WAN v2v 噪声混合 / VACE 等）
    AUX_LATENT = "aux_latent"

    # VACE 视频控制上下文（VACEPreprocessNode 输出 → GeneratorNode 输入）
    VACE_CONTEXT = "vace_context"

    # 图片 latent（VAEEncodeImagesNode 输出）
    IMG_LATENTS = "img_latents"
