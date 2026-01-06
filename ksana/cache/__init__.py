from ..config.cache_config import (
    CustomStepCacheConfig,
    DBCacheConfig,
    DCacheConfig,
    EasyCacheConfig,
    KsanaCacheConfig,
    KsanaHybridCacheConfig,
    MagCacheConfig,
    TeaCacheConfig,
)
from ..models.model_key import KsanaModelKey
from .base_cache import KsanaCache, KsanaHybridCache
from .custom_cache import CustomStepCache
from .dbcache import DBCache
from .dcache import DCache
from .easycache import EasyCache
from .magcache import MagCache
from .teacache import TeaCache


def _create_cache(model_key: KsanaModelKey, config: KsanaCacheConfig):
    if config is None:
        return None
    if isinstance(config, DCacheConfig):
        return DCache(model_key, config)
    elif isinstance(config, CustomStepCacheConfig):
        return CustomStepCache(model_key, config)
    elif isinstance(config, TeaCacheConfig):
        return TeaCache(model_key, config)
    elif isinstance(config, EasyCacheConfig):
        return EasyCache(model_key, config)
    elif isinstance(config, MagCacheConfig):
        return MagCache(model_key, config)
    elif isinstance(config, DBCacheConfig):
        return DBCache(model_key, config)
    else:
        raise ValueError(f"Unknown cache config type: {type(config)}")


def create_hybrid_cache(
    model_key: KsanaModelKey,
    cache_config: KsanaHybridCacheConfig,
):
    return KsanaHybridCache(
        model_key=model_key,
        step_cache=_create_cache(model_key, cache_config.step_cache),
        block_cache=_create_cache(model_key, cache_config.block_cache),
    )


__all__ = [
    "create_hybrid_cache",
    "KsanaCache",
    "KsanaHybridCache",
    "DCache",
    "CustomStepCache",
    "TeaCache",
    "EasyCache",
    "MagCache",
    "DBCache",
]
