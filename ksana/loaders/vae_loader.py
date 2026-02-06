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

from ..models import KsanaQwenVAEModel, KsanaWanVAEModel
from ..models.model_key import KsanaModelKey
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import is_file_or_dir, log, time_range


@KsanaUnitFactory.register(
    KsanaUnitType.LOADER, [KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE]
)
class KsanaVaeLoaderUnit(KsanaLoaderUnit):
    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.VAE_WAN2_1: KsanaWanVAEModel,
        KsanaModelKey.VAE_WAN2_2: KsanaWanVAEModel,
        KsanaModelKey.QwenImageVAE: KsanaQwenVAEModel,
    }

    @time_range
    def run(self, model_path: str, device: torch.device, shard_fn=None):  # pylint: disable=unused-variable
        log.info(f"{self.model_key} loadding vae model")
        if not os.path.exists(model_path) or not is_file_or_dir(model_path):
            raise ValueError(f"model_path {model_path} does not exist or is not a file")
        default_settings = load_default_settings(self.model_key)

        model_class = self._MAP_KEY_TO_MODEL_CLASS.get(self.model_key, None)
        if model_class is None:
            raise NotImplementedError(f"load vae model {self.model_key} not supported yet")
        model = model_class(model_key=self.model_key, default_settings=default_settings, device=device)
        model.load(model_path, shard_fn=shard_fn)
        return model
