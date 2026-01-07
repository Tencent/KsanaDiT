import os
import unittest

import torch

from ksana import KsanaEngine
from ksana.config import (
    KsanaDistributedConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

prompts = [
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 123
# TODO: add test TEST_DTYPE
TEST_DTYPE = torch.float16
TEST_SIZE = (720, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 4
TEST_PORT = int(os.environ.get("KSANA_TEST_PORT", 29500))


class TestKsanaGpus(unittest.TestCase):

    def test_simple_gpus(self):
        print("-----------------test_simple_gpus-----------------")
        engine = KsanaEngine.from_models(
            "./Wan2.2-T2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=2, port=TEST_PORT)
        )
        videos = engine.generate(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=True,
            ),
        )
        with self.subTest(msg="bs2 Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6556231379508972, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.44206780195236206, places=TEST_EPS_PLACE)

        videos = engine.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                rope_function="comfy",
                return_frames=True,
                save_video=True,
            ),
        )
        with self.subTest(msg="bs1 Shape Check"):
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean2 = videos.cpu().abs().mean().item()
        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.6555809378623962, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
