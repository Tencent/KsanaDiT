import unittest
from ksana import KsanaGenerator
from ksana.config import (
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaDistributedConfig,
)
import torch

prompts = [
    "缓慢的平移镜头，在外滩边上，有清风吹过。镜头从远到近，女孩在手舞足蹈的跳舞，舞姿非常美丽，镜头从远景到近景，给出了女孩的特写和细节。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 123
# TODO: add test TEST_DTYPE
TEST_DTYPE = torch.float16
TEST_SIZE = (720, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 4


class TestKsanaWanI2VGpus(unittest.TestCase):

    def test_simple_gpus(self):
        print("-----------------test_simple_gpus-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-I2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=2))
        videos = generator.generate_video(
            prompts,
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
        with self.subTest(msg="bs2 Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, TEST_FRAME_NUM, 576, 576])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.603122353553772, places=TEST_EPS_PLACE)
        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.6307958960533142, places=TEST_EPS_PLACE)

        videos = generator.generate_video(
            prompts[0],
            img_path="./examples/images/start_image.png",
            end_img_path="./examples/images/end_image.png",
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
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, 576, 576])
        mean2 = videos.cpu().abs().mean().item()
        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.32316213846206665, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
