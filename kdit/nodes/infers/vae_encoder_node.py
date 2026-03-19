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

"""VAE Encode Nodes — 直接调用 VAE Model 的编码推理。

拆分为两个独立 Node，各自拥有明确的 tensor 契约：
  - VAEEncodeSpatialNode: start_img + end_img → 视频 latent（含 mask，用于 I2V/VACE）
  - VAEEncodeImagesNode:  image → 图片 latent（用于 Edit/帧级编码）
"""

from kdit.models.model_key import ModelKey
from kdit.tensor import TensorKey
from kdit.utils import log

from ..core.base_node import InferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import InferNodeType, NodeDispatchPolicy


@InferNodeFactory.register(
    InferNodeType.VAE_ENCODE_SPATIAL,
    [ModelKey.VAE_WAN2_1, ModelKey.VAE_WAN2_2, ModelKey.QwenImageVAE],
)
class VAEEncodeSpatialNode(InferNode):
    """VAE 时序条件编码 — rank 0 执行后 broadcast 到所有卡。

    构建视频帧序列（首帧 + 零帧 + 尾帧）→ encode → 返回 (latent, mask)。
    用于 I2V（首尾帧控制）、VACE（视频控制）等场景。
    输出 BASE_LATENT: list[latent, mask] 或 list[latent]（无 mask 时）。
    """

    dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
    input_tensor_keys = [TensorKey.START_IMG, TensorKey.END_IMG]
    output_tensor_keys = [TensorKey.BASE_LATENT]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        vae_model = model_pool.get_model(model_key)
        meta = context.metadata

        start_img = self._get_data(tensor_pool, TensorKey.START_IMG)
        end_img = self._get_data(tensor_pool, TensorKey.END_IMG)
        batch_size = meta.get("batch_size", 1 if start_img is None else start_img.shape[0])

        if start_img is not None and batch_size % start_img.shape[0] != 0:
            raise ValueError(f"start_img batch size {start_img.shape[0]} cannot be broadcast to {batch_size}")

        log.info(
            f"vae_encode with model_key: {vae_model.model_key}, target_batch_size: {batch_size}, "
            f"start_image shape: {start_img.shape if start_img is not None else None}, "
            f"end_image shape: {end_img.shape if end_img is not None else None}, "
            f"mask shape: {meta.get('mask').shape if meta.get('mask') is not None else None}"
        )

        latent, mask = vae_model.forward_encode(
            meta.get("target_f"),
            meta.get("target_h"),
            meta.get("target_w"),
            device=device_ctx.device,
            target_batch_size=batch_size,
            start_img=start_img,
            end_img=end_img,
            mask=meta.get("mask"),
        )

        if latent is not None:
            base_latent_data = [latent, mask] if mask is not None else [latent]
            tensor_pool.put(TensorKey.BASE_LATENT, base_latent_data)


@InferNodeFactory.register(
    InferNodeType.VAE_ENCODE_IMAGES,
    [ModelKey.VAE_WAN2_1, ModelKey.VAE_WAN2_2, ModelKey.QwenImageVAE],
)
class VAEEncodeImagesNode(InferNode):
    """VAE 图片编码 — rank 0 执行后 broadcast 到所有卡。

    将参考图（单张或多张）编码为 latent，纯空间编码，不涉及时序。
    用于 Edit（图片编辑参考图）、VACE（帧级编码）等场景。
    """

    dispatch_policy = NodeDispatchPolicy.R0_R0_BCAST
    input_tensor_keys = [TensorKey.IMAGE]
    output_tensor_keys = [TensorKey.IMAGE_EMBEDS]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        vae_model = model_pool.get_model(model_key)
        meta = context.metadata

        image = self._get_data(tensor_pool, TensorKey.IMAGE)
        batch_size = meta.get("batch_size", 1)

        log.info(
            f"vae_encode_image with model_key: {vae_model.model_key}, "
            f"image type: {type(image).__name__}, batch_size: {batch_size}"
        )

        image_embeds = vae_model.forward_encode_image(
            image=image,
            device=device_ctx.device,
            target_batch_size=batch_size,
        )

        if image_embeds is not None:
            tensor_pool.put(TensorKey.IMAGE_EMBEDS, image_embeds)
