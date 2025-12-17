from dataclasses import dataclass, field

SUPPORTED_CACHE_METHODS = ["DCache", "TeaCache", "EasyCache", "MagCache", "DBCache"]


@dataclass
class KsanaCacheConfig:
    name: str = field(default="KsanaCache")
    offload: bool = field(default=False)


@dataclass
class DCacheConfig(KsanaCacheConfig):
    name: str = field(default="DCache")
    fast_degree: int | None = field(default=None)
    slow_degree: int | None = field(default=None)
    fast_force_calc_every_n_step: int | None = field(default=None)
    slow_force_calc_every_n_step: int | None = field(default=None)
    skip_first_n_iter: int = field(default=2)


@dataclass
class TeaCacheConfig(KsanaCacheConfig):
    name: str = field(default="TeaCache")
    rel_l1_thresh: float | None = field(default=None)
    cache_device: str | None = field(default=None)
    start_step: int | None = field(default=None)
    end_step: int | None = field(default=None)
    use_coeffecients: bool | None = field(default=None)
    mode: str | None = field(default=None)


@dataclass
class EasyCacheConfig(KsanaCacheConfig):
    name: str = field(default="EasyCache")
    reuse_thresh: float | None = field(default=None)
    start_percent: float | None = field(default=None)
    end_percent: float | None = field(default=None)
    verbose: bool | None = field(default=None)


@dataclass
class MagCacheConfig(KsanaCacheConfig):
    name: str = field(default="MagCache")
    threshold: float | None = field(default=None)
    K: int | None = field(default=None)
    cache_device: str | None = field(default=None)
    start_step: int | None = field(default=None)
    end_step: int | None = field(default=None)
    verbose: bool | None = field(default=None)


@dataclass
class DBCacheConfig(KsanaCacheConfig):
    name: str = field(default="DBCache")
    Fn_compute_blocks: int = field(default=8)
    Bn_compute_blocks: int = field(default=0)
    residual_diff_threshold: float = field(default=0.08)
    max_warmup_steps: int = field(default=8)
    warmup_interval: int = field(default=1)
    max_cached_steps: int = field(default=-1)
    max_continuous_cached_steps: int = field(default=-1)
    enable_separate_cfg: bool = field(default=True)
    cfg_compute_first: bool = field(default=False)
    enable_taylorseer: bool = field(default=False)
    taylorseer_order: int = field(default=1)
    num_blocks: int | None = field(default=None)
