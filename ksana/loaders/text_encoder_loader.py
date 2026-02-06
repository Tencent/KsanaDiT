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
