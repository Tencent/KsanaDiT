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

import os
import unittest

from nodes_test_helper import (
    COMFY_MODEL_DIFFUSION_ROOT,
    iter_test_models,
    run_load_and_generate,
)

from ksana.config import (
    DBCacheConfig,
    DCacheConfig,
    EasyCacheConfig,
    KsanaHybridCacheConfig,
    MagCacheConfig,
    TeaCacheConfig,
)
from ksana.models import KsanaModelKey
from ksana.utils.distribute import get_rank_id

TEST_STEPS = 10
EPS_PLACES = 2
CACHE_CONFIGS = [
    DCacheConfig(),
    DBCacheConfig(),
    KsanaHybridCacheConfig(step_cache=DCacheConfig(), block_cache=DBCacheConfig()),
]

CACHE_TEST_CONFIGS = {
    "DBCache": DBCacheConfig(
        fn_compute_blocks=1,
        bn_compute_blocks=0,
        residual_diff_threshold=0.08,
        max_warmup_steps=4,
        warmup_interval=1,
        max_cached_steps=8,
        max_continuous_cached_steps=2,
        enable_separate_cfg=True,
        num_blocks=40,
    ),
    "EasyCache": EasyCacheConfig(
        reuse_thresh=0.06,
        start_percent=0.2,
        end_percent=0.98,
        mode="t2v",
    ),
    "MagCache": MagCacheConfig(
        threshold=0.04,
        max_skip_steps=2,
        retention_ratio=0.2,
        mode="t2v",
    ),
    "TeaCache": TeaCacheConfig(
        threshold=0.2,
        mode="t2v",
        start_step=0,
    ),
}

CACHE_TEST_EXPECTED_MEANS = {
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "DBCache"): 0.77099609375,
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "EasyCache"): 0.7690431314,
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "MagCache"): 0.7695312572,
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", "TeaCache"): 0.76904296875,
}


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
        for model_name, img_shape, text_shape, expected_model_key in iter_test_models():
            if expected_model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.QwenImage_T2I]:
                print(f"-----------------skip {expected_model_key}, cache not supported yet -----------------")
                continue
            for cache_config in CACHE_CONFIGS:
                print(f"-----------------test {model_name} {cache_config} -----------------")
                with self.subTest(msg=f"test {model_name} with {cache_config}"):
                    self.run_once(model_name, img_shape, text_shape, expected_model_key, cache_config)


class TestWan22T2VCachesRegression(unittest.TestCase):

    def test_all_caches(self):
        for model_name, img_shape, text_shape, model_key in iter_test_models():
            if model_key != KsanaModelKey.Wan2_2_T2V_14B:
                continue
            if "fp8" in model_name:
                continue
            for cache_name, cache_config in CACHE_TEST_CONFIGS.items():
                with self.subTest(model=model_name, cache=cache_name):
                    load_output, gen_output = run_load_and_generate(
                        os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
                        img_shape,
                        text_shape,
                        TEST_STEPS,
                        cache_config=cache_config,
                    )
                    self.assertEqual(load_output.model, KsanaModelKey.Wan2_2_T2V_14B)
                    samples = gen_output.samples
                    self.assertIsNotNone(samples)
                    mean_val = samples.cpu().abs().mean().item()
                    expected = CACHE_TEST_EXPECTED_MEANS.get((model_name, cache_name))
                    self.assertAlmostEqual(
                        mean_val,
                        expected,
                        places=EPS_PLACES,
                        msg=f"{model_name} + {cache_name} mean: got {mean_val}, expected {expected}",
                    )


if __name__ == "__main__":
    unittest.main()
