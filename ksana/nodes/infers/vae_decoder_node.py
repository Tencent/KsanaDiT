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

"""VAEDecodeNode — 封装 KsanaVaeDecoder Unit 的解码推理。

从 tensor_pool 读取 latents，调用 VAE Decoder 解码为视频/图像，
结果写入 tensor_pool。
"""

from ksana.models.model_key import KsanaModelKey
from ksana.tensor import TensorKey
from ksana.units import KsanaUnitFactory, KsanaUnitType

from ..core.base_node import KsanaInferNode
from ..core.node_factory import KsanaInferNodeFactory
from ..core.node_types import KsanaDispatchPolicy, KsanaInferNodeType


@KsanaInferNodeFactory.register(
    KsanaInferNodeType.VAE_DECODE,
    [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE],
)
class VAEDecodeNode(KsanaInferNode):
    """VAE 解码 — 只在 rank 0 执行，不广播。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_R0_R0
    input_tensor_keys = [TensorKey.LATENTS]
    output_tensor_keys = [TensorKey.VIDEO]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        latents = self._get_data(tensor_pool, TensorKey.LATENTS)
        vae_model = model_pool.get_model(model_key)
        vae_decoder = KsanaUnitFactory.create(KsanaUnitType.DECODER, model_key)

        video = vae_decoder.run(
            vae_model,
            latents=latents,
            local_rank=0,
            device=device_ctx.device,
            offload_device=device_ctx.offload_device,
            offload_model=context.metadata.get("offload_model", False),
            with_end_image=context.metadata.get("with_end_image", False),
        )

        tensor_pool.put(TensorKey.VIDEO, video)
