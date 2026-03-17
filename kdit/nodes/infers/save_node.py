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

"""SaveNode — 将 tensor_pool 中的视频/图像保存到文件。

仅在 local Pipeline 模式下使用，ComfyUI 模式有自己的输出机制。
SaveNode 不需要模型，注册时 model_key=None。
"""

import os

from kdit.tensor import TensorKey
from kdit.utils import log
from kdit.utils.media import save_image, save_video

from ..core.base_node import InferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import InferNodeType, NodeDispatchPolicy


@InferNodeFactory.register(InferNodeType.SAVE_VIDEO, [None])
class SaveVideoNode(InferNode):
    """保存视频 — 只在 rank 0 执行，不广播。"""

    dispatch_policy = NodeDispatchPolicy.ALL_R0_R0
    input_tensor_keys = [TensorKey.VIDEO]
    output_tensor_keys = []

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        video = self._get_data(tensor_pool, TensorKey.VIDEO)
        if video is None:
            log.warning("SaveVideoNode: no VIDEO tensor found in pool, skipping save.")
            return

        meta = context.metadata
        save_path = meta.get("save_path")
        if not save_path:
            log.info("SaveVideoNode: no save_path in metadata, skipping save.")
            return

        fps = meta.get("fps", 30)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_video(
            tensor=video[None],
            save_file=save_path,
            fps=fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        log.info(f"SaveVideoNode: saved video to {save_path}")


@InferNodeFactory.register(InferNodeType.SAVE_IMAGE, [None])
class SaveImageNode(InferNode):
    """保存图像 — 只在 rank 0 执行，不广播。"""

    dispatch_policy = NodeDispatchPolicy.ALL_R0_R0
    input_tensor_keys = [TensorKey.VIDEO]  # 复用 VIDEO key（图像也存在此 key）
    output_tensor_keys = []

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        image = self._get_data(tensor_pool, TensorKey.VIDEO)
        if image is None:
            log.warning("SaveImageNode: no VIDEO tensor found in pool, skipping save.")
            return

        meta = context.metadata
        save_path = meta.get("save_path")
        if not save_path:
            log.info("SaveImageNode: no save_path in metadata, skipping save.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(tensor=image, path=save_path)
        log.info(f"SaveImageNode: saved image to {save_path}")
