# Copyright 2026 Tencent
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
from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ksana.models.model_key import KsanaModelKey
from pipeline_test_helper import (
    SEED,
    TEST_PORT,
    TEST_STEPS,
    get_platform_config_or_skip,
)

TEST_DTYPE = torch.bfloat16
TEST_SIZE = (512, 512)  # (W, H)
TEST_EPS_PLACE = 1

IMG_PATHS = ["examples/images/woman.png", "examples/images/man.png"]
PROMPT = "the woman and man are hugging together"
NEGATIVE_PROMPT = "blur, bad anatomy, deformed face"


class TestKsanaQwenImageEdit(unittest.TestCase):
    def _create_pipeline(self):
        return KsanaPipeline.from_models(
            "./Qwen-Image-Edit-2511",
            model_config=KsanaModelConfig(
                run_dtype=TEST_DTYPE,
                attention_config=KsanaAttentionConfig(),
            ),
            pipeline_key=KsanaModelKey.QwenImage_Edit,
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
            offload_device="cpu",
        )

    def test_image_edit(self):
        """Multi-ref edit: two reference images (woman + man)."""
        print("-----------------qwen_image_edit test_image_edit-----------------")
        config = {
            "GPU": {"mean": 0.667571902275085},
            "NPU": {"mean": 0.6485475897789001},
        }
        expected = get_platform_config_or_skip(config, test_name="qwen_image_edit.test_image_edit")

        pipeline = self._create_pipeline()

        images = pipeline.generate(
            PROMPT,
            prompt_negative=NEGATIVE_PROMPT,
            img_path=IMG_PATHS,
            sample_config=KsanaSampleConfig(
                steps=TEST_STEPS,
                cfg_scale=4.0,
                solver=KsanaSolverType.FLOWMATCH_EULER,
            ),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                return_frames=True,
                save_output=True,
                offload_model=True,
            ),
        )

        self.assertIsInstance(images, torch.Tensor)
        self.assertGreaterEqual(images.ndim, 4)
        image = images[0]

        mean = image.detach().float().abs().mean().item()
        self.assertAlmostEqual(mean, expected["mean"], places=TEST_EPS_PLACE)

    def test_single_ref_image_edit(self):
        """Single-ref edit: one reference image."""
        print("-----------------qwen_image_edit test_single_ref_image_edit-----------------")
        config = {
            "GPU": {"mean": 0.6605427265167236},
            "NPU": {"mean": 0.6609228849411011},
        }
        expected = get_platform_config_or_skip(config, test_name="qwen_image_edit.test_single_ref_image_edit")

        pipeline = self._create_pipeline()

        images = pipeline.generate(
            PROMPT,
            prompt_negative=NEGATIVE_PROMPT,
            img_path=["examples/images/woman.png"],
            sample_config=KsanaSampleConfig(
                steps=TEST_STEPS,
                cfg_scale=4.0,
                solver=KsanaSolverType.FLOWMATCH_EULER,
            ),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                return_frames=True,
                save_output=True,
                offload_model=True,
            ),
        )

        self.assertIsInstance(images, torch.Tensor)
        self.assertGreaterEqual(images.ndim, 4)
        image = images[0]

        mean = image.detach().float().abs().mean().item()
        self.assertAlmostEqual(mean, expected["mean"], places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
