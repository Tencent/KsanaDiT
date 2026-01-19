import unittest

import torch

from ksana import KsanaPipeline
from ksana.config import (
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaTorchCompileConfig,
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
TEST_EPS_PLACE = 7

RADIAL_TEST_SIZE = (1280, 768)  # should be divisible by block_size
RADIAL_TEST_FRAME_NUM = 33


class TestKsanaPipelineWanT2V(unittest.TestCase):
    def test_simple(self):
        print("-----------------test_simple-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B")
        videos = pipeline.generate(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6672356128692627, places=5)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.37940242886543274, places=5)

    def test_batch_size_per_prompt(self):
        print("-----------------test_batch_size_per_prompt-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B")
        batch_size_per_prompts = [2, 3]
        videos = pipeline.generate(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
                batch_size_per_prompts=[2, 3],
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(
                list(videos.shape), [sum(batch_size_per_prompts), 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]]
            )
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()

        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6556181311607361, places=5)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.652587354183197, places=5)

    def test_larger_seq_batch(self):
        print("-----------------test_larger_seq_batch-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B")
        videos = pipeline.generate(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=(1280, 720),
                frame_num=81,
                return_frames=True,
                save_output=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, 81, 720, 1280])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.5184422135353088, places=5)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.22495749592781067, places=5)

    def test_torch_compile(self):
        print("-----------------test_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B", model_config=model_config)

        video = pipeline.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                rope_function="comfy",
                return_frames=True,
                save_output=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6674289107322693, places=4)


if __name__ == "__main__":
    unittest.main()
