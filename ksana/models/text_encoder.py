import os
from abc import ABC

import torch

from .wan import T5EncoderModel


class KsanaT5Encoder(ABC):
    def __init__(self, model_config, checkpoint_dir, shard_fn):
        _default_config = model_config
        self.device = torch.device("cpu")

        self.model = T5EncoderModel(
            text_len=_default_config.text_len,
            dtype=_default_config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, _default_config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, _default_config.t5_tokenizer),
            shard_fn=shard_fn,
        )

    def forward(self, text):
        # TODO: or other device
        return self.model(text, device=torch.device("cpu"))

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device
