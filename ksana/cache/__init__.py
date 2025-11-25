from .cache_config import (
    KsanaCacheConfig,
    DCacheConfig,
    TeaCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
    SUPPORTED_CACHE_METHODS,
)

from .base_cache import KsanaCache
from .dcache import DCache
from .teacache import TeaCache
from .easycache import EasyCache
from .magcache import MagCache


def create_cache(model_name: str, model_type: str, model_size: str, config: KsanaCacheConfig):
    if config is None:
        return None
    if isinstance(config, DCacheConfig):
        return DCache(model_name, model_type, model_size, config)
    elif isinstance(config, TeaCacheConfig):
        return TeaCache(model_name, model_type, model_size, config)
    elif isinstance(config, EasyCacheConfig):
        return EasyCache(model_name, model_type, model_size, config)
    elif isinstance(config, MagCacheConfig):
        return MagCache(model_name, model_type, model_size, config)
    else:
        raise ValueError(f"Unknown cache config type: {type(config)}")


__all__ = ["create_cache", "KsanaCache", "DCache", "TeaCache", "EasyCache", "MagCache", "SUPPORTED_CACHE_METHODS"]
