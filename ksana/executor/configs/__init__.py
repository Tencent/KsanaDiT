from .base import KsanaExecutorConfig
from .wan import WanExecutorConfig, WanLightLoraExecutorConfig

import os

from ...models import KsanaModel, get_default_model_config


def create_executor_config(model_path, lora_dir=None):
    model_name = os.path.basename(model_path)

    model_name, model_type, model_size = KsanaModel.get_model_type(model_name)
    model_config = get_default_model_config(model_name, model_type, model_size)

    # TODO: support other model types
    if model_name == "wan2.2":
        if lora_dir is not None:
            return WanLightLoraExecutorConfig(default_model_config=model_config)
        else:
            return WanExecutorConfig(default_model_config=model_config)
    else:
        return KsanaExecutorConfig()


__all__ = [
    "KsanaExecutorConfig",
    "WanExecutorConfig",
    "WanLightLoraExecutorConfig",
    "create_executor_config",
]
