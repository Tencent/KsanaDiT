from ksana.config.cache_config import (
    KsanaBlockCacheConfig,
    KsanaStepCacheConfig,
    DCacheConfig,
    CustomStepCacheConfig,
    KsanaHybridCacheConfig,
    TeaCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
    DBCacheConfig,
)


def hybrid_cache(step_cache=None, block_cache=None, name=None):
    if step_cache is None and block_cache is None:
        raise ValueError("step_cache and block_cache cannot be both None")
    if step_cache is not None and not isinstance(step_cache, KsanaStepCacheConfig):
        raise ValueError("step_cache must be KsanaStepCacheConfig")
    if block_cache is not None and not isinstance(block_cache, KsanaBlockCacheConfig):
        raise ValueError("block_cache must be KsanaBlockCacheConfig")
    return KsanaHybridCacheConfig(
        step_cache=step_cache,
        block_cache=block_cache,
        name=name,
    )


def dcache(
    fast_degree,
    slow_degree,
    fast_force_calc_every_n_step,
    slow_force_calc_every_n_step,
    name=None,
    offload=False,
):
    return DCacheConfig(
        name=name,
        fast_degree=fast_degree,
        slow_degree=slow_degree,
        fast_force_calc_every_n_step=fast_force_calc_every_n_step,
        slow_force_calc_every_n_step=slow_force_calc_every_n_step,
        offload=offload,
    )


def custom_step_cache(
    steps,
    scales=1.0,
    name=None,
    offload=False,
):
    print(f"steps: {steps}, scales: {scales}")
    steps = [int(i) for i in steps]
    if isinstance(scales, float):
        scales = [scales] * len(steps)
    return CustomStepCacheConfig(
        steps=steps,
        scales=scales,
        name=name,
        offload=offload,
    )


def teacache(
    rel_l1_thresh,
    cache_device,
    start_step,
    end_step,
    use_coeffecients,
    mode,
    name=None,
):
    return TeaCacheConfig(
        rel_l1_thresh,
        cache_device,
        start_step,
        end_step,
        use_coeffecients,
        mode,
        name=name,
    )


def easy_cache(reuse_thresh, start_percent, end_percent, verbose, name=None):
    return EasyCacheConfig(reuse_thresh, start_percent, end_percent, verbose, name=name)


def mag_cache(threshold, K, cache_device, start_step, end_step, name=None):
    return MagCacheConfig(threshold, K, cache_device, start_step, end_step, name=name)


class KsanaComfyDBCache:
    # TODO(jason): optimize the default values
    PRESETS = {
        "conservative": {
            "Fn_compute_blocks": 10,
            "Bn_compute_blocks": 8,
            "residual_diff_threshold": 0.08,
            "max_warmup_steps": 6,
            "max_cached_steps": -1,
            "max_continuous_cached_steps": -1,
            "enable_taylorseer": False,
            "taylorseer_order": 0,
        },
        "balanced": {
            "Fn_compute_blocks": 8,
            "Bn_compute_blocks": 6,
            "residual_diff_threshold": 0.12,
            "max_warmup_steps": 5,
            "max_cached_steps": -1,
            "max_continuous_cached_steps": -1,
            "enable_taylorseer": True,
            "taylorseer_order": 2,
        },
        "aggressive": {
            "Fn_compute_blocks": 6,
            "Bn_compute_blocks": 4,
            "residual_diff_threshold": 0.18,
            "max_warmup_steps": 4,
            "max_cached_steps": -1,
            "max_continuous_cached_steps": -1,
            "enable_taylorseer": True,
            "taylorseer_order": 3,
            "num_blocks": 40,
        },
        # Wan2.2 specific presets
        "wan22_high": {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "residual_diff_threshold": 0.16,
            "max_warmup_steps": 2,
            "max_cached_steps": 12,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": True,
            "taylorseer_order": 1,
        },
        "wan22_low": {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "residual_diff_threshold": 0.24,
            "max_warmup_steps": 2,
            "max_cached_steps": 30,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": True,
            "taylorseer_order": 1,
        },
    }

    @classmethod
    def dbcache(
        self,
        Fn_compute_blocks=8,
        Bn_compute_blocks=6,
        residual_diff_threshold=0.12,
        max_warmup_steps=5,
        warmup_interval=1,
        max_cached_steps=-1,
        max_continuous_cached_steps=-1,
        enable_separate_cfg=True,
        cfg_compute_first=False,
        enable_taylorseer=True,
        taylorseer_order=1,
        num_blocks=40,
        preset="balanced",
    ):
        if preset != "custom" and preset in self.PRESETS:
            preset_config = self.PRESETS[preset]
            Fn_compute_blocks = preset_config.get("Fn_compute_blocks", Fn_compute_blocks)
            Bn_compute_blocks = preset_config.get("Bn_compute_blocks", Bn_compute_blocks)
            residual_diff_threshold = preset_config.get("residual_diff_threshold", residual_diff_threshold)
            max_warmup_steps = preset_config.get("max_warmup_steps", max_warmup_steps)
            max_cached_steps = preset_config.get("max_cached_steps", max_cached_steps)
            max_continuous_cached_steps = preset_config.get("max_continuous_cached_steps", max_continuous_cached_steps)
            enable_taylorseer = preset_config.get("enable_taylorseer", enable_taylorseer)
            taylorseer_order = preset_config.get("taylorseer_order", taylorseer_order)
            num_blocks = preset_config.get("num_blocks", num_blocks)

        return DBCacheConfig(
            Fn_compute_blocks=Fn_compute_blocks,
            Bn_compute_blocks=Bn_compute_blocks,
            residual_diff_threshold=residual_diff_threshold,
            max_warmup_steps=max_warmup_steps,
            warmup_interval=warmup_interval,
            max_cached_steps=max_cached_steps,
            max_continuous_cached_steps=max_continuous_cached_steps,
            enable_separate_cfg=enable_separate_cfg,
            cfg_compute_first=cfg_compute_first,
            enable_taylorseer=enable_taylorseer,
            taylorseer_order=taylorseer_order,
            num_blocks=num_blocks,
        )
