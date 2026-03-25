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

from kdit.models import KsanaTextEncoderModel
from kdit.models.model_key import ModelKey
from kdit.settings import load_default_settings
from kdit.utils import log, time_profile

from ..core.base_node import IONode
from ..core.node_factory import LoaderNodeFactory
from ..core.node_types import NodeDispatchPolicy


@LoaderNodeFactory.register(
    [ModelKey.T5TextEncoder, ModelKey.Qwen2VLTextEncoder, ModelKey.Qwen2VLTextEncoderMultimodal]
)
class TextEncoderLoaderNode(IONode):
    """加载 TextEncoder 模型。"""

    dispatch_policy = NodeDispatchPolicy.ALL_ALL_ALL

    @time_profile
    def run(self, pins, *, context):
        meta = context.metadata
        model_key = self._factory_model_key
        checkpoint_dir = meta["model_path"]
        log.info(f"{model_key} loading text model")
        if not os.path.exists(checkpoint_dir) or not Path(checkpoint_dir).is_dir():
            raise ValueError(f"checkpoint_dir {checkpoint_dir} should be a directory")

        default_settings = load_default_settings(model_key)
        model = KsanaTextEncoderModel(
            model_key,
            default_settings=default_settings.text_encoder,
            checkpoint_dir=checkpoint_dir,
            device=context.device.offload_device,
            dtype=None,
        )
        pins.put_model(model_key, model)
