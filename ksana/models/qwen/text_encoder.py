"""
Reference (Diffusers):
  - diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
"""

import os
from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from ..base_model import KsanaModel


class KsanaQwen2VLTextEncoderModel(KsanaModel):
    def __init__(
        self,
        checkpoint_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        max_length: int = 1024,
    ):
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self.prompt_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.drop_idx = 34

        tokenizer_dir = os.path.join(checkpoint_dir, "tokenizer")
        if os.path.exists(tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

        text_encoder_dir = os.path.join(checkpoint_dir, "text_encoder")
        if os.path.exists(text_encoder_dir):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                text_encoder_dir, torch_dtype=dtype, trust_remote_code=True
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_dir, torch_dtype=dtype, trust_remote_code=True
            )
        self.model.to(device)
        self.model.eval()

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def forward(
        self,
        prompt: Union[str, List[str]],
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
