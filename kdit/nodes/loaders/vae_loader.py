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

from kdit.models import KsanaQwenVAEModel, KsanaWanVAEModel
from kdit.models.model_key import KsanaModelKey
from kdit.settings import load_default_settings
from kdit.utils import is_file_or_dir, log, time_range

from ..core.base_node import KsanaLoadNode
from ..core.node_factory import LoaderNodeFactory
from ..core.node_types import KsanaDispatchPolicy


@LoaderNodeFactory.register([KsanaModelKey.VAE_WAN2_1, KsanaModelKey.VAE_WAN2_2, KsanaModelKey.QwenImageVAE])
class VAELoaderNode(KsanaLoadNode):
    """加载 VAE 模型。"""

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL

    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.VAE_WAN2_1: KsanaWanVAEModel,
        KsanaModelKey.VAE_WAN2_2: KsanaWanVAEModel,
        KsanaModelKey.QwenImageVAE: KsanaQwenVAEModel,
    }

    @time_range
    def run(self, model_key, *, model_pool, device_ctx, **kwargs):
        model_path = kwargs["model_path"]
        log.info(f"{model_key} loading vae model")
        if not os.path.exists(model_path) or not is_file_or_dir(model_path):
            raise ValueError(f"model_path {model_path} does not exist or is not a file")

        default_settings = load_default_settings(model_key)
        model_class = self._MAP_KEY_TO_MODEL_CLASS.get(model_key)
        if model_class is None:
            raise NotImplementedError(f"load vae model {model_key} not supported yet")

        model = model_class(model_key=model_key, default_settings=default_settings, device=device_ctx.offload_device)
        model.load(model_path, shard_fn=kwargs.get("shard_fn"))
        model_pool.update_model_with_key(model_key, model)
