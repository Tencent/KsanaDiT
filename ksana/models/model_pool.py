from .model_key import KsanaModelKey
from ..utils.utils import singleton
from ..utils.logger import log


@singleton
class KsanaModelPool:
    def __init__(self):
        self.loaded_models = {}

    def update_model(self, model_key: KsanaModelKey, model, allow_exist=False):
        if model_key in self.loaded_models and not allow_exist:
            log.error(f"model_key {model_key} has been loaded")
            raise RuntimeError(f"model_key {model_key} has been loaded")
        self.loaded_models[model_key] = model

    def get_model(self, model_key: KsanaModelKey):
        if model_key is None:
            return None
        if model_key not in self.loaded_models:
            log.error(f"model_key {model_key} has not been loaded")
            raise RuntimeError(f"model_key {model_key} has not been loaded")
        return self.loaded_models.get(model_key)

    def clear(self):
        self.loaded_models.clear()
        self.loaded_models = {}


def get_model_pool() -> KsanaModelPool:
    """
    Get the model_pool instance.
    """
    return KsanaModelPool()
