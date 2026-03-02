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

import ksana.nodes as nodes
from ksana.nodes import KSANA_CACHE_CONFIG, KSANA_CATEGORY_CACHE


class KsanaHybridCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "step_cache": (KSANA_CACHE_CONFIG, {"default": None}),
                "block_cache": (KSANA_CACHE_CONFIG, {"default": None}),
                "name": ("STRING", {"default": "HybridCache"}),
            },
        }

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("hybrid_cache",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.hybrid_cache(*args, **kwargs),)


class KsanaCacheCombineNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "cache": (KSANA_CACHE_CONFIG,),
            },
            "optional": {
                "low_noise_model_cache": (KSANA_CACHE_CONFIG, {"default": None}),
            },
        }

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE
    DESCRIPTION = "Combine Hybrid Caches for 2 models"

    def func(self, cache, low_noise_model_cache=None):
        combined_caches = [cache, low_noise_model_cache] if low_noise_model_cache is not None else [cache]
        return (combined_caches,)


class KsanaDCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "fast_degree": (
                    "FLOAT",
                    {"default": 45, "min": 1, "max": 90, "step": 0.1},
                ),
                "slow_degree": (
                    "FLOAT",
                    {"default": 20, "min": 1, "max": 90, "step": 0.1},
                ),
                "fast_force_calc_every_n_step": ("INT", {"default": 1}),
                "slow_force_calc_every_n_step": ("INT", {"default": 5}),
                "name": ("STRING", {"default": ""}),
                "offload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.dcache(*args, **kwargs),)


class KsanaCustomStepCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "required": {
                "steps": ("FLOAT", {"forceInput": True, "tooltip": "The steps to cache, start from 0"}),
            },
            "optional": {
                "scales": ("FLOAT", {"forceInput": True, "default": 1.0}),
                "name": ("STRING", {"default": ""}),
                "offload": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls):  # pylint: disable=invalid-name
        return True

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.custom_step_cache(*args, **kwargs),)


class KsanaTeaCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "preset": (
                    [
                        "custom",
                        "wan21_t2v",
                        "wan21_i2v_720P",
                        "wan21_i2v_480P",
                        "wan22_t2v",
                        "wan22_i2v",
                        "fast",
                        "balanced",
                        "quality",
                    ],
                    {"default": "balanced"},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.05,
                        "max": 0.5,
                        "step": 0.01,
                    },
                ),
                "mode": (
                    ["t2v_14B", "t2v_1.3B", "i2v_720P", "i2v_480P"],
                    {"default": "t2v_14B"},
                ),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10000, "step": 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": 10000, "step": 1},
                ),
                "cache_device": (
                    ["main_device", "offload_device"],
                    {"default": "main_device"},
                ),
                "verbose": (
                    "BOOLEAN",
                    {"default": False},
                ),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):  # pylint: disable=invalid-name
        return True

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.teacache(*args, **kwargs),)


class KsanaEasyCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "preset": (
                    [
                        "custom",
                        "wan21_t2v",
                        "wan21_i2v",
                        "wan22_t2v",
                        "wan22_i2v",
                        "conservative",
                        "balanced",
                        "aggressive",
                    ],
                    {"default": "balanced"},
                ),
                "reuse_thresh": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "mode": (
                    ["t2v", "i2v"],
                    {"default": "t2v"},
                ),
                "cache_device": (
                    ["main_device", "offload_device"],
                    {"default": "main_device"},
                ),
                "verbose": ("BOOLEAN", {"default": False}),
                "name": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):  # pylint: disable=invalid-name
        return True

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.easy_cache(*args, **kwargs),)


class KsanaMagCacheNode:
    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "preset": (
                    [
                        "custom",
                        "conservative",
                        "balanced",
                        "aggressive",
                        "wan22_t2v",
                        "wan22_i2v",
                    ],
                    {"default": "balanced"},
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.04,
                        "min": 0.001,
                        "max": 0.5,
                        "step": 0.001,
                    },
                ),
                "max_skip_steps": (
                    "INT",
                    {"default": 2, "min": 1, "max": 10, "step": 1},
                ),
                "retention_ratio": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "mode": (
                    ["t2v", "i2v"],
                    {"default": "t2v"},
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
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls):  # pylint: disable=invalid-name
        return True

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.mag_cache(*args, **kwargs),)


class KsanaDBCacheNode:

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "optional": {
                "fn_compute_blocks": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "bn_compute_blocks": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),
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

    RETURN_TYPES = (KSANA_CACHE_CONFIG,)
    RETURN_NAMES = ("cache_config",)
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CACHE

    def func(self, *args, **kwargs):
        return (nodes.KsanaNodeDBCache.dbcache(*args, **kwargs),)
