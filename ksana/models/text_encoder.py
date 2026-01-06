import os

import torch

from .base_model import KsanaModel
from .model_key import KsanaModelKey
from .wan import T5EncoderModel


class KsanaT5Encoder(KsanaModel):
    def __init__(self, default_pipeline_config, checkpoint_dir, shard_fn):
        default_pipeline_config = default_pipeline_config
        self.device = torch.device("cpu")

        self.model = T5EncoderModel(
            text_len=default_pipeline_config.text_len,
            dtype=default_pipeline_config.t5_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir, default_pipeline_config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, default_pipeline_config.t5_tokenizer),
            shard_fn=shard_fn,
        )

    def forward(self, text):
        # TODO: or other device
        return self.model(text, device=torch.device("cpu"))

    def to(self, device):
        if self.device != device:
            self.model.to(device)
        self.device = device

    def get_model_key(self) -> KsanaModelKey:
        return KsanaModelKey.T5TextEncoder
