import os

import torch

from .base_model import KsanaModel
from .model_key import KsanaModelKey
from .qwen import Qwen2VLTextEncoderModel
from .wan import T5EncoderModel


class KsanaTextEncoderModel(KsanaModel):

    _MAP_KEY_TO_CLASS = {
        KsanaModelKey.T5TextEncoder: T5EncoderModel,
        KsanaModelKey.Qwen2VLTextEncoder: Qwen2VLTextEncoderModel,
    }

    def __init__(self, model_key: KsanaModelKey, default_settings, checkpoint_dir, device, dtype):
        super().__init__(model_key, default_settings)
        checkpoint_path = os.path.join(checkpoint_dir, default_settings.checkpoint)
        tokenizer_path = os.path.join(checkpoint_dir, default_settings.tokenizer)

        dtype = dtype or default_settings.dtype
        if dtype is None:
            raise ValueError("dtype should be specified in default_settings")

        text_class = self._MAP_KEY_TO_CLASS.get(self.model_key, None)
        if text_class is None:
            raise ValueError(f"text_encoder {self.model_key} loader not supported yet")

        self.model = text_class(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            text_len=self.default_settings.text_len,
            dtype=dtype,
            device=device,
        )
        self.device = device
        self.dtype = dtype

    def forward(self, text, device=torch.device("cpu")):
        return self.model(text, device=device)

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
