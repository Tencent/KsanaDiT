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

"""VAEDecodeNode — 直接调用 VAE Model 的解码推理。

从 tensor_pool 读取 latents，调用 vae_model.forward_decode() 解码为视频/图像，
结果写入 tensor_pool。
"""

from kdit.models.model_key import ModelKey
from kdit.tensor import TensorKey
from kdit.utils import log

from ..core.base_node import InferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import InferNodeType, NodeDispatchPolicy


@InferNodeFactory.register(
    InferNodeType.VAE_DECODE,
    [ModelKey.VAE_WAN2_1, ModelKey.VAE_WAN2_2, ModelKey.QwenImageVAE],
)
class VAEDecodeNode(InferNode):
    """VAE 解码 — 只在 rank 0 执行，不广播。"""

    dispatch_policy = NodeDispatchPolicy.ALL_R0_R0
    input_defs = [TensorKey.LATENTS]
    output_defs = [TensorKey.VIDEO]

    def run(self, pins, *, context):
        latents = pins.get_tensor(TensorKey.LATENTS)
        vae_model = pins.get_model(self._factory_model_key)
        meta = context.metadata

        video = vae_model.forward_decode(
            latents=latents,
            local_rank=0,
            device=context.device.compute_device,
            with_end_image=meta.get("with_end_image", False),
        )

        if meta.get("offload_model", False) and context.device.offload_device is not None:
            vae_model.to(context.device.offload_device)

        log.info(f"decoder output shape: {video.shape}")
        pins.put_tensor(TensorKey.VIDEO, video)
