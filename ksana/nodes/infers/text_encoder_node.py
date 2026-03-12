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

"""TextEncodeNode — 封装 KsanaBaseTextEncoder Unit 的前向推理。

从 model_pool 获取 text_encoder_model，调用 Unit.run()，
将 positive/negative 写入 tensor_pool。
"""

from ksana.models.model_key import KsanaModelKey
from ksana.tensor import TensorKey
from ksana.units import KsanaUnitFactory, KsanaUnitType

from ..core.base_node import KsanaInferNode
from ..core.node_factory import KsanaInferNodeFactory
from ..core.node_types import KsanaDispatchPolicy, KsanaInferNodeType


@KsanaInferNodeFactory.register(
    KsanaInferNodeType.TEXT_ENCODE,
    [KsanaModelKey.T5TextEncoder, KsanaModelKey.Qwen2VLTextEncoder, KsanaModelKey.Qwen2VLTextEncoderMultimodal],
)
class TextEncodeNode(KsanaInferNode):
    """文本编码 — 每卡独立执行，结果写入 tensor_pool。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL
    input_tensor_keys = []
    output_tensor_keys = [TensorKey.POSITIVE, TensorKey.NEGATIVE]

    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        text_encoder_model = model_pool.get_model(model_key)
        text_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, model_key)

        positive, negative = text_encoder.run(
            text_encoder_model,
            prompts_positive=context.prompt,
            prompts_negative=context.negative_prompt,
            device=context.metadata.get("text_run_device", device_ctx.device),
            offload_device=device_ctx.offload_device,
            offload_model=context.metadata.get("offload_model", False),
            images=context.metadata.get("condition_images"),
        )

        tensor_pool.put(TensorKey.POSITIVE, positive)
        tensor_pool.put(TensorKey.NEGATIVE, negative)
