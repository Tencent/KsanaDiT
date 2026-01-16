import torch

from .base_model import KsanaModel
from .model_key import KsanaModelKey
from .wan import T5EncoderModel


class KsanaT5TextEncoderModel(KsanaModel):
    def __init__(self, model_key, default_settings, checkpoint_path, tokenizer_path, dtype, device, shard_fn=None):
        super().__init__(model_key, default_settings)
        if self.model_key == KsanaModelKey.T5TextEncoder:
            # TODO(rock): maybe support cuda later
            device = torch.device("cpu")
            self.model = T5EncoderModel(
                checkpoint_path=checkpoint_path,
                tokenizer_path=tokenizer_path,
                text_len=self._default_settings.text_len,
                dtype=dtype,
                device=device,
                shard_fn=shard_fn,
            )

        elif self.model_key == KsanaModelKey.Qwen2VLTextEncoder:
            dtype = torch.bfloat16

        self.device = device
        self.dtype = dtype

    def forward(self, text, device=torch.device("cpu")):
        return self.model(text, device=device)

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
