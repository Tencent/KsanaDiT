from ksana.cache.cache_config import (
    DCacheConfig,
    TeaCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
    DBCacheConfig,
)


class KsanaDCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "fast_degree": (
                    "FLOAT",
                    {"default": 70, "min": 1, "max": 90, "step": 0.1},
                ),
                "slow_degree": (
                    "FLOAT",
                    {"default": 35, "min": 1, "max": 90, "step": 0.1},
                ),
                "fast_force_calc_every_n_step": ("INT", {"default": 1}),
                "slow_force_calc_every_n_step": ("INT", {"default": 5}),
                "name": ("STRING", {"default": ""}),
                "offload": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ("KSANA_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = "ksana/cache"

    def func(
        self,
        fast_degree,
        slow_degree,
        fast_force_calc_every_n_step,
        slow_force_calc_every_n_step,
        name=None,
        offload=False,
    ):
        return (
            DCacheConfig(
                name=name,
                fast_degree=fast_degree,
                slow_degree=slow_degree,
                fast_force_calc_every_n_step=fast_force_calc_every_n_step,
                slow_force_calc_every_n_step=slow_force_calc_every_n_step,
                offload=offload,
            ),
        )


class KsanaTeaCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "rel_l1_thresh": (
                    "FLOAT",
                    {"default": 1.000, "min": 0.0001, "max": 1, "step": 0.001},
                ),
                "cache_device": (
                    ["offload_device", "main_device"],
                    {"default": "offload_device"},
                ),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "step": 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": 10000, "step": 1},
                ),
                "use_coeffecients": ("BOOLEAN", {"default": False}),
                "mode": (["e", "e0"], {"default": "e"}),
                "name": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ("KSANA_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = "ksana/cache"

    def func(
        self,
        rel_l1_thresh,
        cache_device,
        start_step,
        end_step,
        use_coeffecients,
        mode,
        name=None,
    ):
        return (
            TeaCacheConfig(
                rel_l1_thresh,
                cache_device,
                start_step,
                end_step,
                use_coeffecients,
                mode,
                name=name,
            ),
        )


class KsanaEasyCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "reuse_thresh": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0001, "max": 1, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ("KSANA_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = "ksana/cache"

    def func(self, reuse_thresh, start_percent, end_percent, verbose, name=None):
        return (EasyCacheConfig(reuse_thresh, start_percent, end_percent, verbose, name=name),)


class KsanaMagCacheNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "threshold": (
                    "FLOAT",
                    {"default": 0.020, "min": 0.0001, "max": 1, "step": 0.001},
                ),
                "K": ("INT", {"default": 4, "min": 0, "max": 100, "step": 1}),
                "cache_device": (
                    ["offload_device", "main_device"],
                    {"default": "offload_device"},
                ),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "step": 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": 10000, "step": 1},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ("KSANA_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = "ksana/cache"

    def func(self, threshold, K, cache_device, start_step, end_step, name=None):
        return (MagCacheConfig(threshold, K, cache_device, start_step, end_step, name=name),)


class KsanaDBCacheNode:
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
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "Fn_compute_blocks": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "Bn_compute_blocks": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),
                "residual_diff_threshold": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_warmup_steps": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "warmup_interval": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "max_cached_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                "max_continuous_cached_steps": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
                "enable_separate_cfg": ("BOOLEAN", {"default": True}),
                "cfg_compute_first": ("BOOLEAN", {"default": False}),
                "enable_taylorseer": ("BOOLEAN", {"default": True}),
                "taylorseer_order": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1}),
                "num_blocks": ("INT", {"default": 40, "min": 1, "max": 200, "step": 1}),
                "preset": (
                    ["custom", "conservative", "balanced", "aggressive", "wan22_high", "wan22_low"],
                    {"default": "balanced"},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):
        return True

    RETURN_TYPES = ("KSANA_CACHE_CONFIG",)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = "ksana/cache"

    def func(
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

        return (
            DBCacheConfig(
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
            ),
        )
