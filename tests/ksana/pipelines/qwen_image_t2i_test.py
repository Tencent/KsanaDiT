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
                batch_size_per_prompts=[1, 2],
            ),
        )

        self.assertEqual(len(images), 3)
        mean0 = images[0].detach().float().mean().item()
        mean1 = images[1].detach().float().mean().item()
        mean2 = images[2].detach().float().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.324187323, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.6177374720573425, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.5726516246795654, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
