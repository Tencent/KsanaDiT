import os

import torch

from ..models import KsanaQwen2VLTextEncoderModel, KsanaT5TextEncoderModel
from ..models.model_key import KsanaModelKey
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType


# Note: 不同模型，加载可以根据模型有不同的加载实现， 但是decoder和encoder，以及Generator应该尽量复用同一个实现
@KsanaUnitFactory.register(KsanaUnitType.LOADER, KsanaModelKey.T5TextEncoder)
class KsanaWanTextEncoderLoaderUnit(KsanaLoaderUnit):
    def run(self, checkpoint_path: str, tokenizer_path: str, device: torch.device, dtype: torch.dtype):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint_path {checkpoint_path} does not exist")
        default_settings = load_default_settings(self.model_key)
        # TODO(rock): maybe support cuda later
        device = torch.device("cpu")
        return KsanaT5TextEncoderModel(
            self.model_key,
            default_settings=default_settings.text_encoder,
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
            dtype=dtype,
        )


@KsanaUnitFactory.register(KsanaUnitType.LOADER, KsanaModelKey.Qwen2VLTextEncoder)
class KsanaQwenTextEncoderLoaderUnit(KsanaLoaderUnit):
    def run(self, checkpoint_path: str, tokenizer_path: str, device: torch.device, dtype: torch.dtype):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint_path {checkpoint_path} does not exist")
        # default_settings = load_default_settings(self.model_key)
        dtype = torch.bfloat16
        return KsanaQwen2VLTextEncoderModel(
            checkpoint_dir=checkpoint_path,
            dtype=dtype,
        )
