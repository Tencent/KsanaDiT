import os

import torch

from ..models.model_key import KsanaModelKey
from ..models.vae_model import KsanaVAEModel
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType


@KsanaUnitFactory.register(
    KsanaUnitType.LOADER, [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE]
)
class KsanaWanVaeLoaderUnit(KsanaLoaderUnit):
    def run(self, model_path: str, device: torch.device, shard_fn=None):  # pylint: disable=unused-variable
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise ValueError(f"model_path {model_path} does not exist or is not a file")
        default_settings = load_default_settings(self.model_key)
        return KsanaVAEModel(
            model_key=self.model_key, model_path=model_path, device=device, default_settings=default_settings
        )
