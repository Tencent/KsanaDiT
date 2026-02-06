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

import unittest

import torch
from pipeline_test_helper import (
    PROMPTS,
    SEED,
    TEST_PORT,
    TEST_STEPS,
    get_platform_config_or_skip,
)

from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

TEST_DTYPE = torch.bfloat16
TEST_SIZE = (512, 512)  # (W, H)
TEST_EPS_PLACE = 2  # 统一精度，兼容单卡/多卡的数值差异


class TestKsanaQwenImageT2I(unittest.TestCase):
    def _assert_image_tensor_ok(self, image: torch.Tensor, *, expected_wh: tuple[int, int]):
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.ndim, 4)  # [B, C, H, W]
        self.assertEqual(list(image.shape[:2]), [1, 3])
        self.assertEqual(list(image.shape[2:]), [expected_wh[1], expected_wh[0]])

    def test_batch_prompts(self):
        print("-----------------qwen_image test_batch_prompts-----------------")
        config = {
            "GPU": {"mean0": 0.3167570531368255, "mean1": 0.6125394105911255, "mean2": 0.5676110982894897},
            "NPU": {"mean0": 0.304778993, "mean1": 0.6123599410057068, "mean2": 0.598247766494751},
        }
        expected = get_platform_config_or_skip(config, test_name="qwen_image.test_batch_prompts")
        generator = KsanaPipeline.from_models(
            "./Qwen-Image",
            model_config=KsanaModelConfig(
                run_dtype=TEST_DTYPE,
                attention_config=KsanaAttentionConfig(),
            ),
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
            offload_device="cpu",
        )
        images = generator.generate(
            PROMPTS,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                return_frames=True,
                save_output=True,
                offload_model=True,
                batch_size_per_prompts=[1, 2],
            ),
        )

        self.assertEqual(len(images), 3)
        mean0 = images[0].detach().float().mean().item()
        mean1 = images[1].detach().float().mean().item()
        mean2 = images[2].detach().float().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, expected["mean0"], places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, expected["mean1"], places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, expected["mean2"], places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
