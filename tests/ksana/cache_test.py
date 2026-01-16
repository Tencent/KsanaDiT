import unittest

import torch

from ksana.cache import (
    CustomStepCache,
    DBCache,
    DCache,
    EasyCache,
    MagCache,
    TeaCache,
)
from ksana.cache.base_cache import KsanaBlockCache
from ksana.config.cache_config import (
    CustomStepCacheConfig,
    DBCacheConfig,
    DCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
    TeaCacheConfig,
)
from ksana.models.model_key import KsanaModelKey


def check_cache_apis(class_type, config_type, model_key=KsanaModelKey.Wan2_2_T2V_14B):
    cache_config = config_type(steps=0) if config_type is CustomStepCacheConfig else config_type()
    cache = class_type(model_key, cache_config)
    x = torch.randn(1, 14, 14, 14)

    if isinstance(cache, KsanaBlockCache):
        x = cache(phase="combine", x=x, step_iter=30, timestep=50, blocks=None)
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{class_type.__name__} __call__ should return torch.Tensor")
        cache.offload_to_cpu()
        cache.show_cache_rate()
        return
    if cache.valid_for(phase="cond", x=x, step_iter=1, timestep=1):
        x = cache(phase="cond", x=x, step_iter=1, timestep=1)
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{class_type.__name__} __call__ should return torch.Tensor")
    else:
        r = cache(phase="cond", x=x, step_iter=1, timestep=1)
        if r is not None:
            raise ValueError(f"{class_type.__name__} __call__ should return None")

    if cache.valid_for(phase="uncond", x=x, step_iter=10, timestep=20):
        x = cache(phase="uncond", x=x, step_iter=10, timestep=20)
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{class_type.__name__} __call__ should return torch.Tensor")
    else:
        r = cache(phase="uncond", x=x, step_iter=10, timestep=20)
        if r is not None:
            raise ValueError(f"{class_type.__name__} __call__ should return None")

    if cache.valid_for(phase="combine", x=x, step_iter=30, timestep=50):
        x = cache(phase="combine", x=x, step_iter=30, timestep=50)
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{class_type.__name__} __call__ should return torch.Tensor")
    else:
        r = cache(phase="combine", x=x, step_iter=30, timestep=50)
        if r is not None:
            raise ValueError(f"{class_type.__name__} __call__ should return None")

    cache.record_input_before_update(phase="combine", x=x, step_iter=30, timestep=50)
    x_output = torch.randn(1, 14, 14, 14)
    cache.update_cache(phase="combine", x=x_output, step_iter=30, timestep=50)

    cache.offload_to_cpu()
    cache.show_cache_rate()


class TestCacheAPIs(unittest.TestCase):
    def test_cache_apis(self):
        test_class = [
            (DCache, DCacheConfig),
            (CustomStepCache, CustomStepCacheConfig),
            (TeaCache, TeaCacheConfig),
            (EasyCache, EasyCacheConfig),
            (MagCache, MagCacheConfig),
            (DBCache, DBCacheConfig),
        ]

        for class_type, config_type in test_class:
            try:
                check_cache_apis(class_type, config_type)
                self.assertTrue(True)
            except Exception as e:  # pylint: disable=broad-except
                self.fail(f"check_cache_apis({class_type.__name__}, {config_type.__name__}) failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
