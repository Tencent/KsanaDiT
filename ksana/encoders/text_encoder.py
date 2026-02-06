# Copyright 2025 Tencent
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

from abc import abstractmethod

import torch
from torch.nn.utils.rnn import pad_sequence

from ..models.model_key import KsanaModelKey
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import str_to_list, time_range


class KsanaBaseTextEncoder(KsanaRunnerUnit):

    def _valid_input_prompt_to_list(self, prompt, target_len=None) -> list:
        prompts = str_to_list(prompt)
        if target_len is not None:
            if len(prompts) == 1:
                prompts = prompts * target_len
            elif len(prompts) != target_len:
                raise ValueError(f"prompt length ({len(prompts)}) must match target length ({target_len})")
        return prompts

    @abstractmethod
    def do_run(self, model, prompts_positive_list, prompts_negative_list):
        pass

    @time_range
    def run(
        self,
        model,
        prompts_positive: str | list[str],
        prompts_negative: str | list[str] | None = None,
        device=None,
        offload_device=None,
        offload_model=False,
    ):
        prompts_positive_list = self._valid_input_prompt_to_list(prompts_positive)
        prompts_negative_list = prompts_negative or getattr(model.default_settings, "neg_prompt", None)
        prompts_negative_list = self._valid_input_prompt_to_list(prompts_negative_list, len(prompts_positive_list))
        if len(prompts_positive_list) != len(prompts_negative_list):
            raise RuntimeError(
                f"The number of negative prompts ({len(prompts_negative_list)}) "
                f"must match the number of positive prompts ({len(prompts_positive_list)})."
            )
        assert device is not None
        if model.device != device:
            model.to(device)

        positive, negative = self.do_run(model, prompts_positive_list, prompts_negative_list, device=device)

        if offload_model and offload_device is not None and offload_device != device:
            model.to(offload_device)

        return positive, negative


# Note: 不同类型的模型应该尽量用同一个encoder实现, 不同模型加载可以根据模型有不同的加载实现
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.T5TextEncoder)
class KsanaTextEncoder(KsanaBaseTextEncoder):
    def do_run(
        self, model, prompts_positive_list: list[str], prompts_negative_list: list[str] | None = None, device=None
    ):
        all_prompts = prompts_positive_list + prompts_negative_list
        all_embeddings_list = model.forward(all_prompts)

        # TODO(qiannan): self.text_encoder.forward tokenizer的时候是填充到相同长度了，
        # 但是返回是裁剪了，所以如果返回不裁剪，就不需要pad了
        # Pad the combined list of tensors to the max length in the entire batch.
        all_padded_embeddings = pad_sequence(all_embeddings_list, batch_first=True, padding_value=0.0)

        # Split the padded tensor back into positive and negative parts.
        positive, negative = torch.chunk(all_padded_embeddings, 2, dim=0)

        return positive, negative


@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.Qwen2VLTextEncoder)
class KsanaQwenVLTextEncoderUnit(KsanaBaseTextEncoder):

    def do_run(
        self,
        model,
        prompts_positive_list: list[str],
        prompts_negative_list: list[str] | None = None,
        device=None,
    ):
        # TODO(qiannan): support batch inference
        positive_embeds, positive_mask = model.forward(prompts_positive_list, device=device)
        negative_embeds, negative_mask = model.forward(prompts_negative_list, device=device)

        return (positive_embeds, positive_mask), (negative_embeds, negative_mask)
