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

"""TextEncodeNode — 文本编码推理节点。

从 model_pool 获取 text_encoder_model，执行编码，
将 positive/negative 写入 tensor_pool。
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from kdit.models.model_key import KsanaModelKey
from kdit.tensor import TensorKey
from kdit.utils import str_to_list, time_range

from ..core.base_node import KsanaInferNode
from ..core.node_factory import KsanaInferNodeFactory
from ..core.node_types import KsanaDispatchPolicy, KsanaInferNodeType


def _validate_prompts(prompt, negative_prompt, default_neg_prompt=None, target_len=None):
    """校验并归一化 prompt 列表。

    Returns:
        (prompts_positive_list, prompts_negative_list)
    """
    prompts_positive_list = str_to_list(prompt)
    if target_len is not None and len(prompts_positive_list) == 1:
        prompts_positive_list = prompts_positive_list * target_len

    prompts_negative_list = negative_prompt or default_neg_prompt
    prompts_negative_list = str_to_list(prompts_negative_list)
    if len(prompts_negative_list) == 1:
        prompts_negative_list = prompts_negative_list * len(prompts_positive_list)

    if len(prompts_positive_list) != len(prompts_negative_list):
        raise RuntimeError(
            f"The number of negative prompts ({len(prompts_negative_list)}) "
            f"must match the number of positive prompts ({len(prompts_positive_list)})."
        )
    return prompts_positive_list, prompts_negative_list


def _offload_model_if_needed(model, offload_model, offload_device, current_device):
    """条件性地将模型卸载到 offload 设备。"""
    if offload_model and offload_device is not None and offload_device != current_device:
        model.to(offload_device)


@KsanaInferNodeFactory.register(KsanaInferNodeType.TEXT_ENCODE, KsanaModelKey.T5TextEncoder)
class T5TextEncodeNode(KsanaInferNode):
    """T5 文本编码 — forward 后 pad + chunk 拆分 pos/neg。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL
    input_tensor_keys = []
    output_tensor_keys = [TensorKey.POSITIVE, TensorKey.NEGATIVE]

    @time_range
    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        model = model_pool.get_model(model_key)
        device = context.metadata.get("text_run_device", device_ctx.device)
        meta = context.metadata

        prompts_positive_list, prompts_negative_list = _validate_prompts(
            context.prompt,
            context.negative_prompt,
            default_neg_prompt=getattr(model.default_settings, "neg_prompt", None),
        )

        assert device is not None
        if model.device != device:
            model.to(device)

        # T5: 合并 pos+neg → forward → pad → chunk 拆分
        all_prompts = prompts_positive_list + prompts_negative_list
        all_embeddings_list = model.forward(all_prompts)

        all_padded_embeddings = pad_sequence(all_embeddings_list, batch_first=True, padding_value=0.0)
        positive, negative = torch.chunk(all_padded_embeddings, 2, dim=0)

        _offload_model_if_needed(
            model,
            offload_model=meta.get("offload_model", False),
            offload_device=device_ctx.offload_device,
            current_device=device,
        )

        tensor_pool.put(TensorKey.POSITIVE, positive)
        tensor_pool.put(TensorKey.NEGATIVE, negative)


@KsanaInferNodeFactory.register(
    KsanaInferNodeType.TEXT_ENCODE,
    [KsanaModelKey.Qwen2VLTextEncoder, KsanaModelKey.Qwen2VLTextEncoderMultimodal],
)
class QwenTextEncodeNode(KsanaInferNode):
    """Qwen VL 文本编码 — 分别 forward pos/neg，返回 (embeds, mask)。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL
    input_tensor_keys = []
    output_tensor_keys = [TensorKey.POSITIVE, TensorKey.NEGATIVE]

    @time_range
    def run(self, model_key, context, *, tensor_pool, model_pool, device_ctx):
        model = model_pool.get_model(model_key)
        device = context.metadata.get("text_run_device", device_ctx.device)
        meta = context.metadata
        images = meta.get("condition_images")

        prompts_positive_list, prompts_negative_list = _validate_prompts(
            context.prompt,
            context.negative_prompt,
            default_neg_prompt=getattr(model.default_settings, "neg_prompt", None),
        )

        assert device is not None
        if model.device != device:
            model.to(device)

        # Qwen: 分别 forward pos/neg
        positive_embeds, positive_mask = model.forward(prompts_positive_list, images=images, device=device)
        negative_embeds, negative_mask = model.forward(prompts_negative_list, images=images, device=device)

        _offload_model_if_needed(
            model,
            offload_model=meta.get("offload_model", False),
            offload_device=device_ctx.offload_device,
            current_device=device,
        )

        tensor_pool.put(TensorKey.POSITIVE, (positive_embeds, positive_mask))
        tensor_pool.put(TensorKey.NEGATIVE, (negative_embeds, negative_mask))
