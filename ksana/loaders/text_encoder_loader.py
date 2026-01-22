import os
from pathlib import Path

import torch

from ..models import KsanaTextEncoderModel
from ..models.model_key import KsanaModelKey
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range


@KsanaUnitFactory.register(KsanaUnitType.LOADER, [KsanaModelKey.Qwen2VLTextEncoder, KsanaModelKey.T5TextEncoder])
class KsanaTextEncoderLoaderUnit(KsanaLoaderUnit):

    @time_range
    def run(self, checkpoint_dir: str, device: torch.device = torch.device("cpu"), dtype: torch.dtype = None):
        log.info(f"{self.model_key} loading text model")
        if not os.path.exists(checkpoint_dir) or not Path(checkpoint_dir).is_dir():
            raise ValueError(f"checkpoint_dir {checkpoint_dir} should be a directory")
        default_settings = load_default_settings(self.model_key)
        return KsanaTextEncoderModel(
            self.model_key,
            default_settings=default_settings.text_encoder,
            checkpoint_dir=checkpoint_dir,
            device=device,
            dtype=dtype,
        )
