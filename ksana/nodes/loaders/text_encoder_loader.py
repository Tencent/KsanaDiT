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

from ksana.models import KsanaTextEncoderModel
from ksana.models.model_key import KsanaModelKey
from ksana.settings import load_default_settings
from ksana.utils import log, time_range

from ..core.base_node import KsanaLoadNode
from ..core.node_factory import KsanaLoaderNodeFactory
from ..core.node_types import KsanaDispatchPolicy


@KsanaLoaderNodeFactory.register(
    [KsanaModelKey.T5TextEncoder, KsanaModelKey.Qwen2VLTextEncoder, KsanaModelKey.Qwen2VLTextEncoderMultimodal]
)
class TextEncoderLoaderNode(KsanaLoadNode):
    """加载 TextEncoder 模型。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL

    @time_range
    def run(self, model_key, *, model_pool, device_ctx, **kwargs):
        checkpoint_dir = kwargs["model_path"]
        log.info(f"{model_key} loading text model")
        if not os.path.exists(checkpoint_dir) or not Path(checkpoint_dir).is_dir():
            raise ValueError(f"checkpoint_dir {checkpoint_dir} should be a directory")

        default_settings = load_default_settings(model_key)
        model = KsanaTextEncoderModel(
            model_key,
            default_settings=default_settings.text_encoder,
            checkpoint_dir=checkpoint_dir,
            device=device_ctx.offload_device,
            dtype=None,
        )
        model_pool.update_model_with_key(model_key, model)
