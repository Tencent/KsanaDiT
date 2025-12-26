import torch
from .base_cache import KsanaStepCache
from ..config.cache_config import TeaCacheConfig
from ..utils import log
from ..models.model_key import KsanaModelKey


class TeaCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: TeaCacheConfig):
        super().__init__(model_key, config)

    def valid_for(self, **kwargs) -> bool:
        log.error("TeaCache valid_for not implemented")
        return False

    def __call__(self, **kwargs) -> torch.Tensor:
        log.error("TeaCache __call__ not implemented")
        return

    def record_input_before_update(self, **kwargs):
        log.error("TeaCache record_input_before_update not implemented")
        return

    def update_cache(self, **kwargs):
        log.error("TeaCache update_cache not implemented")
        return

    def offload_to_cpu(self):
        log.error("TeaCache offload_to_cpu not implemented")
        return

    def show_cache_rate(self):
        log.error("TeaCache show_cache_rate not implemented")
        return
