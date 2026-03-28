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

"""ReadImageNode — 读取图片文件并输出 tensor。

单一功能：读取一个或多个图片路径，拼成一个 tensor 输出。
通过 DAG 多实例 + cross-pin connect 区分用途（start_img / end_img / ref_images）。
"""

import torch
from PIL import Image
from torchvision import transforms

from kdit.tensor import TensorKey
from kdit.utils import log

from ..core.base_node import IONode
from ..core.node_factory import IONodeFactory
from ..core.node_types import IONodeType, NodeDispatchPolicy


def _load_image_paths(img_paths: str | list[str], device: str = "cpu") -> torch.Tensor | None:
    """加载一个或多个图片路径为 tensor。

    Returns:
        Tensor [N, C, H, W] in range [-1, 1]，或 None（路径无效时）。
    """
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    if not img_paths:
        return None

    to_tensor = transforms.ToTensor()
    tensors = []
    for p in img_paths:
        try:
            img = Image.open(p).convert("RGB")
            # to_tensor 输出 [0, 1]，归一化到 [-1, 1] 以匹配 VAE 编码器期望
            tensors.append(to_tensor(img).sub_(0.5).div_(0.5))
        except OSError:
            log.warning(f"ReadImageNode: failed to load image: {p}")
            continue

    if not tensors:
        return None

    # 堆叠为 [N, C, H, W]
    return torch.stack(tensors).to(device)


@IONodeFactory.register(IONodeType.READ_IMAGE, [None])
class ReadImageNode(IONode):
    """读取图片文件并输出 tensor。

    单一功能：读取一个或多个图片路径，拼成一个 tensor 输出。
    通过 DAG 多实例 + cross-pin connect 区分用途。

    metadata 中需要 ``img_paths: str | list[str] | None``。
    """

    dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
    input_defs = []
    output_defs = [TensorKey.IMAGE]

    def run(self, pins, *, context):
        meta = context.metadata
        img_paths = meta.get("img_paths")  # str | list[str] | None
        if img_paths is None:
            return

        device = context.device.compute_device if context.device else "cpu"
        img_tensor = _load_image_paths(img_paths, device=device)
        if img_tensor is not None:
            log.info(f"ReadImageNode: loaded {img_tensor.shape[0]} image(s), shape={img_tensor.shape}")
            pins.put_tensor(TensorKey.IMAGE, img_tensor)
