import unittest

import torch

from ksana import KsanaEngine
from ksana.config import (
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

QWEN_IMAGE_DIR = "./Qwen-Image"

prompts = [
    "街头摄影，纽约街头，涂鸦墙背景，戴耳机的酷女孩滑板，动态姿势，黄金时刻光线，主体清晰背景虚化，超写实。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实。",
]

SEED = 123
TEST_DTYPE = torch.bfloat16
TEST_SIZE = (512, 512)  # (W, H)
TEST_STEPS = 5
TEST_EPS_PLACE = 7


class TestKsanaQwenImageT2I(unittest.TestCase):
    def _assert_image_tensor_ok(self, image: torch.Tensor, *, expected_wh: tuple[int, int]):
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.ndim, 4)  # [B, C, H, W]
        self.assertEqual(list(image.shape[:2]), [1, 3])
        self.assertEqual(list(image.shape[2:]), [expected_wh[1], expected_wh[0]])

    def test_simple(self):
        print("-----------------qwen_image test_simple-----------------")
        generator = KsanaEngine.from_models(
            QWEN_IMAGE_DIR,
            model_config=KsanaModelConfig(run_dtype=TEST_DTYPE),
            offload_device="cpu",
        )
        image = generator.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                return_frames=True,
                save_output=False,
                offload_model=True,
            ),
        )
        self._assert_image_tensor_ok(image, expected_wh=TEST_SIZE)
        mean = image.detach().float().mean().item()
        self.assertAlmostEqual(mean, 0.29857462644577026, places=TEST_EPS_PLACE)

    def test_batch_prompts(self):
        print("-----------------qwen_image test_batch_prompts-----------------")
        generator = KsanaEngine.from_models(
            QWEN_IMAGE_DIR,
            model_config=KsanaModelConfig(run_dtype=TEST_DTYPE),
            offload_device="cpu",
        )
        images = generator.generate(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                return_frames=True,
                save_output=True,
                offload_model=True,
            ),
        )

        self.assertEqual(len(images), len(prompts))
        mean0 = images[0].detach().float().mean().item()
        mean1 = images[1].detach().float().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.29761189222335815, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.5015290975570679, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
