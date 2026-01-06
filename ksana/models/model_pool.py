import gc

import torch

from ..utils.logger import log
from .base_model import KsanaModel
from .model_key import KsanaModelKey


class KsanaModelPool:
    def __init__(self):
        self.loaded_models: dict[KsanaModelKey, KsanaModel] = {}

    def update_model(self, model: KsanaModel, allow_exist=False):
        model_key = model.get_model_key()
        if model_key in self.loaded_models:
            log.warning(f"model_key {model_key} has been loaded")
            if not allow_exist:
                raise RuntimeError(f"model_key {model_key} has been loaded")
        log.info(f"loaded model {model_key}")
        self.loaded_models[model_key] = model

    def update_models(self, model_list: list[KsanaModel], allow_exist=False):
        if not isinstance(model_list, (tuple, list)):
            model_list = [model_list]
        for model in model_list:
            self.update_model(model, allow_exist)

    def get_model(self, model_key: KsanaModelKey) -> KsanaModel:
        if model_key is None:
            return None
        if model_key not in self.loaded_models:
            log.error(f"model_key {model_key} has not been loaded")
            raise RuntimeError(f"model_key {model_key} has not been loaded")
        return self.loaded_models.get(model_key)

    def get_models(self, model_key_list: list[KsanaModelKey] | tuple[KsanaModelKey, ...]) -> list[KsanaModel]:
        if model_key_list is None:
            return []
        return [self.get_model(model_key) for model_key in model_key_list]

    def clear_models(self, model_keys: list[KsanaModelKey] | KsanaModelKey = None):
        """clear models loaded by this executor
        clear all if model_keys is None
        """
        if model_keys is None:
            log.info("clear all models")
            self.loaded_models.clear()
            self.loaded_models = {}
            gc.collect()
            torch.cuda.empty_cache()
            return

        if not isinstance(model_keys, (list, tuple)):
            model_keys = [model_keys]
        for one_model_key in model_keys:
            self.loaded_models.pop(one_model_key)
            log.info(f"clear {one_model_key}")
        gc.collect()
        torch.cuda.empty_cache()
