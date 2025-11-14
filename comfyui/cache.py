from ksana.cache.cache_config import (
    DCacheConfig,
    TeaCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
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
            }
            # "optional": {
            #     "low_fast_degree": ("FLOAT", {"default": 50, "min": 1, "max": 90, "step": 0.1}),
            #     "low_slow_degree": ("FLOAT", {"default": 30, "min": 1, "max": 90, "step": 0.1}),
            #     "low_fast_n_step": ("INT", {"default": 2}),
            #     "low_slow_n_step": ("INT", {"default": 4}),
            #     "skip_first_n_iter": ("INT", {"default": 2})
            #     },
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
    ):
        return (
            DCacheConfig(
                fast_degree,
                slow_degree,
                fast_force_calc_every_n_step,
                slow_force_calc_every_n_step,
                name=name,
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
