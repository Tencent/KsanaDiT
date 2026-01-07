import unittest

import torch

from ksana import KsanaEngine
from ksana.config import (
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

prompts = [
    "缓慢的平移镜头，在外滩边上，有清风吹过。镜头从远到近，女孩在手舞足蹈的跳舞，舞姿非常美丽，镜头从远景到近景，给出了女孩的特写和细节。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 123
TEST_DTYPE = torch.float16
TEST_SIZE = (720, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 7


class TestKsanaI2V(unittest.TestCase):

    def test_simple_i2v(self):
        print("-----------------test_simple_i2v-----------------")
        engine = KsanaEngine.from_models("./Wan2.2-I2V-A14B")
        videos = engine.generate(
            prompts[0],
            img_path="./examples/images/input.png",
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=True,
            ),
        )
        with self.subTest(msg="bs1 Shape Check"):
            # 576 is from image shape and target shape
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, 576, 576])
        mean0 = videos.cpu().abs().mean().item()

        videos = engine.generate(
            prompts,
            img_path="./examples/images/start_image.png",
            end_img_path="./examples/images/end_image.png",
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                rope_function="comfy",
                save_video=True,
            ),
        )
        mean1 = videos[0].cpu().abs().mean().item()
        mean2 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="bs 2 Shape Check"):
            self.assertEqual(list(videos.shape), [2, 3, TEST_FRAME_NUM, 576, 576])
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6031375527381897, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.4757797420024872, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.49633023142814636, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
