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

"""VAE Encode Nodes — 封装 KsanaVaeEncoder Unit 的前向推理。

拆分为两个独立 Node，各自拥有明确的 tensor 契约：
  - VAEEncodeSpatialNode: start_img + end_img → 视频 latent（含 mask，用于 I2V/VACE）
  - VAEEncodeImagesNode:  image → 图片 latent（用于 Edit/帧级编码）
"""

from ksana.models.model_key import KsanaModelKey
from ksana.tensor import KsanaTensorKey
from ksana.units import KsanaUnitFactory, KsanaUnitType

from ..core.base_node import KsanaInferNode
from ..core.node_factory import KsanaInferNodeFactory
from ..core.node_types import KsanaDispatchPolicy, KsanaInferNodeType


@KsanaInferNodeFactory.register(
    KsanaInferNodeType.VAE_ENCODE_SPATIAL,
    [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE],
)
class VAEEncodeSpatialNode(KsanaInferNode):
    """VAE 时序条件编码 — rank 0 执行后 broadcast 到所有卡。

    构建视频帧序列（首帧 + 零帧 + 尾帧）→ encode → 拼接 mask 通道。
    用于 I2V（首尾帧控制）、VACE（视频控制）等场景。
    """

    dispatch_policy = KsanaDispatchPolicy.R0_R0_BCAST
    input_tensor_keys = [KsanaTensorKey.START_IMG, KsanaTensorKey.END_IMG]
    output_tensor_keys = [KsanaTensorKey.IMAGE_EMBEDS]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        vae_model = model_pool.get_model(model_key)
        vae_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, model_key)
        meta = context.metadata

        image_embeds = vae_encoder.run(
            vae_model,
            start_img=tensor_pool.get(KsanaTensorKey.START_IMG),
            end_img=tensor_pool.get(KsanaTensorKey.END_IMG),
            mask=meta.get("mask"),
            batch_size=meta.get("batch_size"),
            target_f=meta.get("target_f"),
            target_h=meta.get("target_h"),
            target_w=meta.get("target_w"),
            device=device_ctx.device,
        )

        if image_embeds is not None:
            # 统一为 list[Tensor]，I2V 场景包装为单元素 list
            if not isinstance(image_embeds, list):
                image_embeds = [image_embeds]
            tensor_pool.put(KsanaTensorKey.IMAGE_EMBEDS, image_embeds)


@KsanaInferNodeFactory.register(
    KsanaInferNodeType.VAE_ENCODE_IMAGES,
    [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE],
)
class VAEEncodeImagesNode(KsanaInferNode):
    """VAE 图片编码 — rank 0 执行后 broadcast 到所有卡。

    将参考图（单张或多张）编码为 latent，纯空间编码，不涉及时序。
    用于 Edit（图片编辑参考图）、VACE（帧级编码）等场景。
    """

    dispatch_policy = KsanaDispatchPolicy.R0_R0_BCAST
    input_tensor_keys = [KsanaTensorKey.IMAGE]
    output_tensor_keys = [KsanaTensorKey.IMAGE_EMBEDS]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        vae_model = model_pool.get_model(model_key)
        vae_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, model_key)
        meta = context.metadata

        image_embeds = vae_encoder.run_encode_image(
            vae_model,
            image=tensor_pool.get(KsanaTensorKey.IMAGE),
            device=device_ctx.device,
            batch_size=meta.get("batch_size", 1),
        )

        if image_embeds is not None:
            # 统一为 list[Tensor]，单 tensor 包装为单元素 list
            if not isinstance(image_embeds, list):
                image_embeds = [image_embeds]
            tensor_pool.put(KsanaTensorKey.IMAGE_EMBEDS, image_embeds)
