import unittest
import torch

from ksana.config.cache_config import (
    DCacheConfig,
    CustomStepCacheConfig,
    TeaCacheConfig,
    EasyCacheConfig,
    MagCacheConfig,
    DBCacheConfig,
)

from ksana.cache import (
    DCache,
    CustomStepCache,
    TeaCache,
    EasyCache,
    MagCache,
    DBCache,
)

from ksana.cache.base_cache import KsanaBlockCache

from ksana.models.model_key import KsanaModelKey


def check_cache_apis(class_type, config_type, model_key=KsanaModelKey.Wan2_2_T2V_14B_HIGH):
    cache = class_type(model_key, config_type)
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
            except Exception as e:
                self.fail(f"check_cache_apis({class_type.__name__}, {config_type.__name__}) failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
