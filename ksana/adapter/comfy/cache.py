# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ksana.config.cache_config import (
    CustomStepCacheConfig,
    DBCacheConfig,
    DCacheConfig,
    EasyCacheConfig,
    KsanaBlockCacheConfig,
    KsanaHybridCacheConfig,
    KsanaStepCacheConfig,
    MagCacheConfig,
    TeaCacheConfig,
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


class KsanaNodeTeaCache:
    PRESETS = {
        "wan21_t2v": {"threshold": 0.2, "mode": "t2v_14B"},
        "wan21_t2v_1.3B": {"threshold": 0.2, "mode": "t2v_1.3B"},
        "wan21_i2v_720P": {"threshold": 0.2, "mode": "i2v_720P"},
        "wan21_i2v_480P": {"threshold": 0.2, "mode": "i2v_480P"},
        "wan22_t2v": {"threshold": 0.2, "mode": "t2v_14B"},
        "wan22_i2v": {"threshold": 0.2, "mode": "i2v_720P"},
        "fast": {"threshold": 0.3, "mode": "t2v_14B"},
        "balanced": {"threshold": 0.2, "mode": "t2v_14B"},
        "quality": {"threshold": 0.1, "mode": "t2v_14B"},
    }

    @classmethod
    def teacache(
        cls,
        threshold=0.2,
        mode="t2v_14B",
        start_step=0,
        end_step=-1,
        cache_device=None,
        verbose=False,
        preset="balanced",
        name=None,
        **kwargs,
    ):
        if preset != "custom" and preset in cls.PRESETS:
            preset_config = cls.PRESETS[preset]
            threshold = preset_config.get("threshold", threshold)
            mode = preset_config.get("mode", mode)

        device = None if cache_device == "main_device" else "cpu" if cache_device == "offload_device" else cache_device
        return TeaCacheConfig(
            name=name,
            threshold=threshold,
            mode=mode,
            start_step=start_step,
            end_step=end_step if end_step >= 0 else None,
            cache_device=device,
            verbose=verbose,
        )


def teacache(*args, **kwargs):
    return KsanaNodeTeaCache.teacache(*args, **kwargs)


class KsanaNodeEasyCache:
    PRESETS = {
        "wan21_t2v": {
            "reuse_thresh": 0.05,
            "start_percent": 0.2,
            "end_percent": 0.98,
            "mode": "t2v",
        },
        "wan21_i2v": {
            "reuse_thresh": 0.05,
            "start_percent": 0.2,
            "end_percent": 0.98,
            "mode": "i2v",
        },
        "wan22_t2v": {
            "reuse_thresh": 0.06,
            "start_percent": 0.2,
            "end_percent": 0.98,
            "mode": "t2v",
        },
        "wan22_i2v": {
            "reuse_thresh": 0.05,
            "start_percent": 0.2,
            "end_percent": 0.98,
            "mode": "i2v",
        },
        "conservative": {
            "reuse_thresh": 0.03,
            "start_percent": 0.25,
            "end_percent": 0.95,
            "mode": "t2v",
        },
        "balanced": {
            "reuse_thresh": 0.05,
            "start_percent": 0.2,
            "end_percent": 0.98,
            "mode": "t2v",
        },
        "aggressive": {
            "reuse_thresh": 0.1,
            "start_percent": 0.15,
            "end_percent": 0.98,
            "mode": "t2v",
        },
    }

    @classmethod
    def easy_cache(
        cls,
        reuse_thresh=0.05,
        start_percent=0.2,
        end_percent=0.98,
        mode="t2v",
        cache_device=None,
        verbose=False,
        preset="balanced",
        name=None,
    ):
        if preset != "custom" and preset in cls.PRESETS:
            preset_config = cls.PRESETS[preset]
            reuse_thresh = preset_config.get("reuse_thresh", reuse_thresh)
            start_percent = preset_config.get("start_percent", start_percent)
            end_percent = preset_config.get("end_percent", end_percent)
            mode = preset_config.get("mode", mode)

        device = None if cache_device == "main_device" else "cpu" if cache_device == "offload_device" else cache_device
        return EasyCacheConfig(
            name=name,
            reuse_thresh=reuse_thresh,
            start_percent=start_percent,
            end_percent=end_percent,
            mode=mode,
            cache_device=device,
            verbose=verbose,
        )


def easy_cache(*args, **kwargs):
    return KsanaNodeEasyCache.easy_cache(*args, **kwargs)


class KsanaNodeMagCache:
    PRESETS = {
        "conservative": {
            "threshold": 0.02,
            "k": 1,
            "retention_ratio": 0.3,
        },
        "balanced": {
            "threshold": 0.04,
            "k": 2,
            "retention_ratio": 0.2,
        },
        "aggressive": {
            "threshold": 0.08,
            "k": 3,
            "retention_ratio": 0.15,
        },
        "wan22_t2v": {
            "threshold": 0.04,
            "k": 2,
            "retention_ratio": 0.2,
            "mode": "t2v",
        },
        "wan22_i2v": {
            "threshold": 0.04,
            "k": 2,
            "retention_ratio": 0.2,
            "mode": "i2v",
        },
    }

    @classmethod
    def mag_cache(
        cls,
        threshold=0.04,
        max_skip_steps=2,
        k=None,
        retention_ratio=0.2,
        mode="t2v",
        cache_device=None,
        start_step=0,
        end_step=-1,
        verbose=False,
        preset="balanced",
        name=None,
    ):
        if preset != "custom" and preset in cls.PRESETS:
            preset_config = cls.PRESETS[preset]
            threshold = preset_config.get("threshold", threshold)
            max_skip_steps = preset_config.get("k", max_skip_steps)
            retention_ratio = preset_config.get("retention_ratio", retention_ratio)
            mode = preset_config.get("mode", mode)

        if k is not None:
            max_skip_steps = k
        device = None if cache_device == "main_device" else "cpu"
        return MagCacheConfig(
            name=name,
            threshold=threshold,
            max_skip_steps=max_skip_steps,
            k=k,
            retention_ratio=retention_ratio,
            mode=mode,
            cache_device=device,
            start_step=start_step,
            end_step=end_step if end_step >= 0 else None,
            verbose=verbose,
        )


def mag_cache(*args, **kwargs):
    return KsanaNodeMagCache.mag_cache(*args, **kwargs)


class KsanaNodeDBCache:
    # TODO(jason): optimize the default values
    PRESETS = {
        "conservative": {
            "fn_compute_blocks": 10,
            "bn_compute_blocks": 8,
            "residual_diff_threshold": 0.08,
            "max_warmup_steps": 6,
            "max_cached_steps": -1,
            "max_continuous_cached_steps": -1,
            "enable_taylorseer": False,
            "taylorseer_order": 0,
        },
        "balanced": {
            "fn_compute_blocks": 8,
            "bn_compute_blocks": 6,
            "residual_diff_threshold": 0.12,
            "max_warmup_steps": 5,
            "max_cached_steps": -1,
            "max_continuous_cached_steps": -1,
            "enable_taylorseer": True,
            "taylorseer_order": 2,
        },
        "aggressive": {
            "fn_compute_blocks": 6,
            "bn_compute_blocks": 4,
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
            "fn_compute_blocks": 1,
            "bn_compute_blocks": 0,
            "residual_diff_threshold": 0.16,
            "max_warmup_steps": 2,
            "max_cached_steps": 12,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": True,
            "taylorseer_order": 1,
        },
        "wan22_low": {
            "fn_compute_blocks": 1,
            "bn_compute_blocks": 0,
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
        fn_compute_blocks=8,
        bn_compute_blocks=6,
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
            fn_compute_blocks = preset_config.get("fn_compute_blocks", fn_compute_blocks)
            bn_compute_blocks = preset_config.get("bn_compute_blocks", bn_compute_blocks)
            residual_diff_threshold = preset_config.get("residual_diff_threshold", residual_diff_threshold)
            max_warmup_steps = preset_config.get("max_warmup_steps", max_warmup_steps)
            max_cached_steps = preset_config.get("max_cached_steps", max_cached_steps)
            max_continuous_cached_steps = preset_config.get("max_continuous_cached_steps", max_continuous_cached_steps)
            enable_taylorseer = preset_config.get("enable_taylorseer", enable_taylorseer)
            taylorseer_order = preset_config.get("taylorseer_order", taylorseer_order)
            num_blocks = preset_config.get("num_blocks", num_blocks)

        return DBCacheConfig(
            fn_compute_blocks=fn_compute_blocks,
            bn_compute_blocks=bn_compute_blocks,
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
