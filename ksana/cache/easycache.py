from .base_cache import KsanaStepCache
from ..config.cache_config import EasyCacheConfig
from ..models.model_key import KsanaModelKey
from ..utils import log
import torch


class EasyCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: EasyCacheConfig):
        super().__init__(model_key, config)

    def valid_for(self, **kwargs) -> bool:
        log.error("EasyCache valid_for not implemented")
        return False

    def __call__(self, **kwargs) -> torch.Tensor:
        log.error("EasyCache __call__ not implemented")
        return

    def record_input_before_update(self, **kwargs):
        log.error("EasyCache record_input_before_update not implemented")
        return

    def update_cache(self, **kwargs):
        log.error("EasyCache update_cache not implemented")
        return

    def offload_to_cpu(self):
        log.error("EasyCache offload_to_cpu not implemented")
        return

    def show_cache_rate(self):
        log.error("EasyCache show_cache_rate not implemented")
        return
