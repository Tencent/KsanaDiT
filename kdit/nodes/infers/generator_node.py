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

"""GeneratorNode — 封装 Diffusion Generator Unit 的去噪推理。

从 tensor_pool 读取 positive/negative/image_embeds，
调用 Generator Unit 执行去噪，将 latents 无条件写入 tensor_pool。
"""

from kdit.generators import GeneratorFactory
from kdit.generators.generator_context import GeneratorInferContext
from kdit.models.model_key import KsanaModelKey
from kdit.tensor import TensorKey

from ..core.base_node import KsanaInferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import KsanaDispatchPolicy, KsanaInferNodeType


@InferNodeFactory.register(
    KsanaInferNodeType.GENERATE,
    [
        KsanaModelKey.Wan2_2_T2V_14B,
        KsanaModelKey.Wan2_2_I2V_14B,
        KsanaModelKey.Wan2_1_VACE_14B,
        KsanaModelKey.QwenImage_T2I,
        KsanaModelKey.QwenImage_Edit,
    ],
)
class GeneratorNode(KsanaInferNode):
    """Diffusion 去噪 — 所有卡并行执行。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL
    input_tensor_keys = [
        TensorKey.POSITIVE,
        TensorKey.NEGATIVE,
        TensorKey.IMAGE_EMBEDS,
        TensorKey.INPUT_LATENT,
    ]
    output_tensor_keys = [TensorKey.LATENTS]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        image_embeds = self._get_data(tensor_pool, TensorKey.IMAGE_EMBEDS)  # list[Tensor] | None
        meta = context.metadata

        noise_shape = meta.get("noise_shape")
        if noise_shape is None and image_embeds is not None and len(image_embeds) > 0:
            noise_shape = list(image_embeds[0].shape[1:])

        ctx = GeneratorInferContext(
            diffusion_model=model_pool.get_model(model_key),
            positive=self._get_data(tensor_pool, TensorKey.POSITIVE),
            negative=self._get_data(tensor_pool, TensorKey.NEGATIVE),
            image_embeds=image_embeds,
            input_latent=self._get_data(tensor_pool, TensorKey.INPUT_LATENT),
            noise_shape=noise_shape,
            device=device_ctx.device,
            offload_device=device_ctx.offload_device,
            sample_config=context.sample_config,
            runtime_config=context.runtime_config,
            cache_config=context.cache_config,
            video_control=meta.get("video_control"),
            control_video_config=meta.get("control_video_config"),
            comfy_bar_callback=meta.get("comfy_bar_callback"),
        )

        generator = GeneratorFactory.create(model_key)
        latents = generator.run(ctx)
        tensor_pool.put(TensorKey.LATENTS, latents)
