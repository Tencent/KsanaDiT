import unittest

import torch

from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionBackend,
    KsanaAttentionConfig,
    KsanaLoraConfig,
    KsanaModelConfig,
    KsanaRadialSageAttentionConfig,
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
TEST_EPS_PLACE = 4

RADIAL_TEST_SIZE = (1280, 768)  # should be divisible by block_size
RADIAL_TEST_FRAME_NUM = 33


class TestKsanaPipelineWan22T2VLora(unittest.TestCase):
    def test_lora(self):
        print("-----------------test_lora and radial_sage_attn-----------------")
        radial_sage_attn_config = KsanaRadialSageAttentionConfig(
            dense_blocks_num=20,
            dense_attn_steps=0,
            decay_factor=0.2,
            block_size=64,
            dense_attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
        )

        model_config = KsanaModelConfig(
            attention_config=radial_sage_attn_config,
        )

        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            lora_config=KsanaLoraConfig("./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"),
            model_config=model_config,
        )

        video = pipeline.generate(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=RADIAL_TEST_SIZE,
                frame_num=RADIAL_TEST_FRAME_NUM,
                return_frames=True,
                save_output=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, RADIAL_TEST_FRAME_NUM, RADIAL_TEST_SIZE[1], RADIAL_TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.20027998089790344, places=TEST_EPS_PLACE)

    def test_lora_torch_compile(self):
        print("-----------------test_lora_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            lora_config=KsanaLoraConfig("./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"),
            model_config=model_config,
        )
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
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.2548159062862396, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
