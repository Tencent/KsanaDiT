from dataclasses import dataclass, field


@dataclass
class KsanaCacheConfig:
    name: str = field(default="KsanaCache")
    offload: bool = field(default=False)


@dataclass
class KsanaBlockCacheConfig(KsanaCacheConfig):
    name: str = field(default="KsanaBlockCache")


@dataclass
class KsanaStepCacheConfig(KsanaCacheConfig):
    name: str = field(default="KsanaStepCache")


@dataclass
class KsanaHybridCacheConfig:
    name: str = field(default="KsanaHybridCache")
    step_cache: KsanaStepCacheConfig | None = None
    block_cache: KsanaBlockCacheConfig | None = None


def warp_as_hybrid_cache(
    cache_config: KsanaCacheConfig,
) -> KsanaHybridCacheConfig:
    if isinstance(cache_config, KsanaHybridCacheConfig):
        return cache_config
    if cache_config is None or not isinstance(cache_config, KsanaCacheConfig):
        raise ValueError("cache_config must be provided")
    out = KsanaHybridCacheConfig()
    if isinstance(cache_config, KsanaBlockCacheConfig):
        out.block_cache = cache_config
        out.name = f"Hybrid_{cache_config.name}"
    elif isinstance(cache_config, KsanaStepCacheConfig):
        out.step_cache = cache_config
        out.name = f"Hybrid_{cache_config.name}"
    else:
        raise ValueError(
            f"cache_config must be KsanaBlockCacheConfig or KsanaStepCacheConfig, but got {type(cache_config)}"
        )
    return out
