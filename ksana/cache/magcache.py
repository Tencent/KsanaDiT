import torch

from ..config.cache_config import MagCacheConfig
from ..models.model_key import KsanaModelKey
from ..utils import log
from .base_cache import KsanaStepCache


class MagCache(KsanaStepCache):
    def __init__(self, model_key: KsanaModelKey, config: MagCacheConfig):
        super().__init__(model_key, config)

    def valid_for(self, **kwargs) -> bool:
        log.error("MagCache valid_for not implemented")
        return False

    def __call__(self, **kwargs) -> torch.Tensor:
        log.error("MagCache __call__ not implemented")
        return

    def record_input_before_update(self, **kwargs):
        log.error("MagCache record_input_before_update not implemented")
        return

    def update_cache(self, **kwargs):
        log.error("MagCache update_cache not implemented")
        return

    def offload_to_cpu(self):
        log.error("MagCache offload_to_cpu not implemented")
        return

    def show_cache_rate(self):
        log.error("MagCache show_cache_rate not implemented")
        return
