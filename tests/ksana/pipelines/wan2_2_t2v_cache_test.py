import unittest

import torch

from ksana import KsanaPipeline
from ksana.config import (
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)
from ksana.config.cache_config import DBCacheConfig, DCacheConfig, KsanaHybridCacheConfig

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


class TestKsanaPipelineWanT2VCache(unittest.TestCase):

    def test_cache(self):
        # TODO: step 1 can not test cache, real test cache logical,
        print("-----------------test_cache-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B")
        video = pipeline.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=False,
            ),
            cache_config=DCacheConfig(),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6660090088844299, places=TEST_EPS_PLACE)
        video = pipeline.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=False,
            ),
            cache_config=KsanaHybridCacheConfig(step_cache=DCacheConfig(), block_cache=DBCacheConfig()),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        self.assertAlmostEqual(mean, 0.6660090088844299, places=TEST_EPS_PLACE)
        video = pipeline.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=False,
            ),
            cache_config=[DCacheConfig(fast_degree=55), DCacheConfig(fast_degree=45)],
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])


if __name__ == "__main__":
    unittest.main()
