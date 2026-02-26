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

import os

import torch

from .base_model import KsanaModel
from .model_key import KsanaModelKey
from .qwen import Qwen2VLTextEncoderModel
from .qwen.multimodal_text_encoder import Qwen2VLMultimodalTextEncoderModel
from .wan import T5EncoderModel


class KsanaTextEncoderModel(KsanaModel):

    _MAP_KEY_TO_CLASS = {
        KsanaModelKey.T5TextEncoder: T5EncoderModel,
        KsanaModelKey.Qwen2VLTextEncoder: Qwen2VLTextEncoderModel,
        KsanaModelKey.Qwen2VLTextEncoderMultimodal: Qwen2VLMultimodalTextEncoderModel,
    }

    def __init__(self, model_key: KsanaModelKey, default_settings, checkpoint_dir, device, dtype):
        super().__init__(model_key, default_settings)
        checkpoint_path = os.path.join(checkpoint_dir, default_settings.checkpoint)
        # 配置中使用 tokenizer 字段（可以是 tokenizer 或 processor 路径）
        tokenizer_path = os.path.join(checkpoint_dir, default_settings.tokenizer)

        dtype = dtype or default_settings.dtype
        if dtype is None:
            raise ValueError("dtype should be specified in default_settings")

        text_class = self._MAP_KEY_TO_CLASS.get(self.model_key, None)
        if text_class is None:
            raise ValueError(f"text_encoder {self.model_key} loader not supported yet")

        extra_kwargs = {}
        if hasattr(self.default_settings, "prompt_template_drop_idx"):
            extra_kwargs["prompt_template_drop_idx"] = self.default_settings.prompt_template_drop_idx

        self.model = text_class(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            text_len=self.default_settings.text_len,
            dtype=dtype,
            device=device,
            **extra_kwargs,
        )
        self.device = device
        self.dtype = dtype

    def forward(self, text, images=None, device=torch.device("cpu")):
        # Note: 多模态编码器需要传入 images
        if self.model_key == KsanaModelKey.Qwen2VLTextEncoderMultimodal:
            return self.model(text, images=images, device=device)
        return self.model(text, device=device)

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
