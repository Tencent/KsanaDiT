import os

import torch

from ..models import KsanaQwenVAEModel, KsanaWanVAEModel
from ..models.model_key import KsanaModelKey
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import is_file_or_dir


@KsanaUnitFactory.register(
    KsanaUnitType.LOADER, [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE]
)
class KsanaVaeLoaderUnit(KsanaLoaderUnit):
    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.VAE_WAN2_1: KsanaWanVAEModel,
        KsanaModelKey.VAE_WAN2_2: KsanaWanVAEModel,
        KsanaModelKey.QwenImageVAE: KsanaQwenVAEModel,
    }

    def run(self, model_path: str, device: torch.device, shard_fn=None):  # pylint: disable=unused-variable
        if not os.path.exists(model_path) or not is_file_or_dir(model_path):
            raise ValueError(f"model_path {model_path} does not exist or is not a file")
        default_settings = load_default_settings(self.model_key)

        model_class = self._MAP_KEY_TO_MODEL_CLASS.get(self.model_key, None)
        if model_class is None:
            raise NotImplementedError(f"load vae model {self.model_key} not supported yet")
        model = model_class(model_key=self.model_key, default_settings=default_settings, device=device)
        model.load(model_path, shard_fn=shard_fn)
        return model
