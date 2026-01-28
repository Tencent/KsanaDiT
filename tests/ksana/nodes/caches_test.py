import os
import unittest

from nodes_test_helper import COMFY_MODEL_DIFFUSION_ROOT, TEST_MODELS, run_load_and_generate

from ksana.config import DBCacheConfig, DCacheConfig, KsanaHybridCacheConfig
from ksana.models import KsanaModelKey
from ksana.utils.distribute import get_rank_id

TEST_STEPS = 10

CACHE_CONFIGS = [
    DCacheConfig(),
    DBCacheConfig(),
    KsanaHybridCacheConfig(step_cache=DCacheConfig(), block_cache=DBCacheConfig()),
]


class TestCacheAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, cache_config):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            TEST_STEPS,
            cache_config=cache_config,
        )
        self.assertEqual(load_output.model, expected_model_key)

        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def test_all_cache_configs(self):
        for model_name, img_shape, text_shape, expected_model_key in TEST_MODELS:
            if expected_model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.QwenImage_T2I]:
                print(f"-----------------skip {expected_model_key}, cache not supported yet -----------------")
                continue
            for cache_config in CACHE_CONFIGS:
                print(f"-----------------test {model_name} {cache_config} -----------------")
                with self.subTest(msg=f"test {model_name} with {cache_config}"):
                    self.run_once(model_name, img_shape, text_shape, expected_model_key, cache_config)


if __name__ == "__main__":
    unittest.main()
