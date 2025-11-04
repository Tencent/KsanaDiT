from .cache_config import vDitCacheConfig, DCacheConfig, TeaCacheConfig, EasyCacheConfig, MagCacheConfig
from .base_cache import vDitCache

from .dcache import DCache
from .teacache import TeaCache
from .easycache import EasyCache
from .magcache import MagCache


def create_cache(model_kind:str, model_type:str, model_size:str, config: vDitCacheConfig):
    if config is None:
        return None
    if isinstance(config, DCacheConfig):
        return DCache(model_kind, model_type, model_size, config)
    elif isinstance(config, TeaCacheConfig):
        return TeaCache(model_kind, model_type, model_size, config)
    elif isinstance(config, EasyCacheConfig):
        return EasyCache(model_kind, model_type, model_size, config)
    elif isinstance(config, MagCacheConfig):
        return MagCache(model_kind, model_type, model_size, config)
    else:
        raise ValueError(f"Unknown cache config type: {type(config)}")

__all__ = [
    "create_cache"
    "vDitCache",
    "DCache",
    "TeaCache",
    "EasyCache",
    "MagCache"
]