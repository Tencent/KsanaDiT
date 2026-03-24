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

从 tensor_pool 读取 positive/negative/base_latent/aux_latent，
调用 Generator Unit 执行去噪，将 latents 无条件写入 tensor_pool。
"""

from kdit.generators.generator_context import AuxLatent, BaseLatent, GeneratorInferContext
from kdit.generators.generator_def import get_generator_def
from kdit.generators.generator_runner import GeneratorRunner
from kdit.models.model_key import ModelKey
from kdit.tensor import TensorKey

from ..core.base_node import InferNode
from ..core.node_factory import InferNodeFactory
from ..core.node_types import InferNodeType, NodeDispatchPolicy


@InferNodeFactory.register(
    InferNodeType.GENERATE,
    [
        ModelKey.Wan2_2_T2V_14B,
        ModelKey.Wan2_2_I2V_14B,
        ModelKey.Wan2_1_VACE_14B,
        ModelKey.QwenImage_T2I,
        ModelKey.QwenImage_Edit,
    ],
)
class GeneratorNode(InferNode):
    """Diffusion 去噪 — 所有卡并行执行。"""

    dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL
    input_tensor_pins = [
        TensorKey.POSITIVE,
        TensorKey.NEGATIVE,
        TensorKey.BASE_LATENT,
        TensorKey.AUX_LATENT,
        TensorKey.VACE_CONTEXT,
    ]
    output_tensor_pins = [TensorKey.LATENTS]

    def run(self, pins, *, context):
        model_key = self.input_model_pins[0]
        base_latent_data = pins.get_tensor(TensorKey.BASE_LATENT)  # list[Tensor] | None
        aux_latent_data = pins.get_tensor(TensorKey.AUX_LATENT)  # Tensor | list[Tensor] | None
        vace_context_data = pins.get_tensor(TensorKey.VACE_CONTEXT)  # Tensor | None
        meta = context.metadata

        # 从 base_latent_data 构建 BaseLatent 对象 — base_latent 现在是必须的
        if base_latent_data is None or len(base_latent_data) == 0:
            raise ValueError(
                "GeneratorNode requires BASE_LATENT in tensor_pool. "
                "Ensure VAE_COMPUTE_SHAPE or VAE_ENCODE_SPATIAL runs before GENERATE."
            )
        latent = base_latent_data[0]
        mask = base_latent_data[1] if len(base_latent_data) > 1 else None
        base_latent = BaseLatent(latent=latent, mask=mask)

        # 从 aux_latent_data 构建 AuxLatent 对象
        aux_latent = None
        if aux_latent_data is not None:
            aux_latent = AuxLatent(latent=aux_latent_data)

        # VACE_CONTEXT pin 优先于 metadata 中的 control_video_config
        control_video_config = vace_context_data if vace_context_data is not None else meta.get("control_video_config")

        ctx = GeneratorInferContext(
            diffusion_model=pins.get_model(model_key),
            positive=pins.get_tensor(TensorKey.POSITIVE),
            negative=pins.get_tensor(TensorKey.NEGATIVE),
            base_latent=base_latent,
            aux_latent=aux_latent,
            device=context.device.device,
            offload_device=context.device.offload_device,
            sample_config=context.sample_config,
            runtime_config=context.runtime_config,
            cache_config=context.cache_config,
            video_control=meta.get("video_control"),
            control_video_config=control_video_config,
            comfy_bar_callback=meta.get("comfy_bar_callback"),
        )

        gen_def = get_generator_def(model_key)
        runner = GeneratorRunner(gen_def)
        latents = runner.run(ctx)
        pins.put_tensor(TensorKey.LATENTS, latents)
