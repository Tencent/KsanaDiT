import unittest

import torch
from pipeline_test_helper import PROMPTS, SEED, TEST_PORT, TEST_STEPS

from ksana import KsanaPipeline
from ksana.config import (
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)
from ksana.utils.distribute import get_gpu_count

TEST_DTYPE = torch.bfloat16
TEST_SIZE = (512, 512)  # (W, H)
TEST_EPS_PLACE = 2 if get_gpu_count() != 1 else 3


class TestKsanaQwenImageT2I(unittest.TestCase):
    def _assert_image_tensor_ok(self, image: torch.Tensor, *, expected_wh: tuple[int, int]):
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.ndim, 4)  # [B, C, H, W]
        self.assertEqual(list(image.shape[:2]), [1, 3])
        self.assertEqual(list(image.shape[2:]), [expected_wh[1], expected_wh[0]])

    def test_batch_prompts(self):
        print("-----------------qwen_image test_batch_prompts-----------------")
        generator = KsanaPipeline.from_models(
            "./Qwen-Image",
            model_config=KsanaModelConfig(run_dtype=TEST_DTYPE),
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
            ),
        )

        self.assertEqual(len(images), len(PROMPTS))
        mean0 = images[0].detach().float().mean().item()
        mean1 = images[1].detach().float().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.325140208, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.617037892, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
