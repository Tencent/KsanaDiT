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

"""
Reference (Diffusers):
  - diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
"""

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


class Qwen2VLTextEncoderModel:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        text_len: int = 1024,
    ):
        self.device = device
        self.dtype = dtype
        self.max_length = text_len

        self.prompt_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.drop_idx = 34

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=dtype, trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def __call__(
        self,
        prompt: str | list[str],
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        txt = [self.prompt_template.format(p) for p in prompt]

        txt_tokens = self.tokenizer(
            txt,
            max_length=self.max_length + self.drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]

        split_hidden = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden = [h[self.drop_idx :] for h in split_hidden]

        attn_masks = [torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in split_hidden]

        max_seq_len = max(h.size(0) for h in split_hidden)
        prompt_embeds = torch.stack(
            [torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))]) for h in split_hidden]
        )
        attention_mask = torch.stack([torch.cat([m, m.new_zeros(max_seq_len - m.size(0))]) for m in attn_masks])

        return prompt_embeds.to(device, dtype=self.dtype), attention_mask.to(device)

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
