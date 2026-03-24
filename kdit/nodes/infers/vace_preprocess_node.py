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

"""VACEPreprocessNode — VACE 视频控制预处理。

替代 wan.py 中的 _valid_video_control_config + vae_encode_fn 闭包。
通过 model pin 直接访问 VAE 模型进行编码。
输出 VaceConfig（含 vace_context 编码结果）到 VACE_CONTEXT pin。
"""

from kdit.tensor import TensorKey
from kdit.utils import log
from kdit.utils.vace import VaceConfig, encode_vace_context

from ..core.base_node import InferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import InferNodeType, NodeDispatchPolicy


@InferNodeFactory.register(InferNodeType.VACE_PREPROCESS, [None])
class VACEPreprocessNode(InferNode):
    """VACE 视频控制预处理 — 编码控制视频帧。

    从 metadata 中读取 ``video_control_config: VaceConfig``，
    使用 VAE 模型编码控制视频帧，输出完整的 VaceConfig（含 vace_context）。

    如果 video_control_config 为 None 或无控制内容，则不输出。
    """

    dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
    input_tensor_pins = []
    output_tensor_pins = [TensorKey.VACE_CONTEXT]

    def run(self, pins, *, context):
        meta = context.metadata
        video_control_config: VaceConfig | None = meta.get("video_control_config")
        if video_control_config is None:
            return

        # 需要 VAE 模型来编码控制视频帧
        vae_model_key = self.input_model_pins[0] if self.input_model_pins else None
        if vae_model_key is None:
            log.warning("VACEPreprocessNode: no VAE model pin connected, skipping")
            return

        vae_model = pins.get_model(vae_model_key)
        device = context.device.device if context.device else "cpu"

        # 编码控制视频帧
        encoded_config = encode_vace_context(vae_model, video_control_config, device=device)
        if encoded_config is not None:
            log.info("VACEPreprocessNode: encoded vace context successfully")
            pins.put_tensor(TensorKey.VACE_CONTEXT, encoded_config)
