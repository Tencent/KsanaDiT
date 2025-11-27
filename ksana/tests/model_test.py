# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import unittest
from ksana import KsanaGenerator
from ksana.config import (
    KsanaModelConfig,
    KsanaTorchCompileConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)
import torch

prompt = (
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"
)
SEED = 123
# TODO: add test TEST_DTYPE
TEST_DTYPE = torch.float16
TEST_SIZE = (720, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 7


class TestKsana(unittest.TestCase):

    def test_simple(self):
        print("-----------------test_simple-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=True,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6659602522850037, places=TEST_EPS_PLACE)

    def test_larger_seq(self):
        print("-----------------test_larger_seq-----------------")
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B/high_noise_model.safetensors",
            text_checkpoint_dir="./Wan2.2-T2V-A14B",
            vae_checkpoint_dir="./Wan2.2-T2V-A14B",
        )
        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=1),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=(1280, 720),
                frame_num=81,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6659602522850037, places=TEST_EPS_PLACE)

    def test_cache(self):
        # TODO: step 1 can not test cache, real test cache logical,
        print("-----------------test_cache-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                cache_method="DCache",
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6659602522850037, places=TEST_EPS_PLACE)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B", lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"
        )
        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.25350645184516907, places=TEST_EPS_PLACE)

    def test_torch_compile(self):
        print("-----------------test_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B", model_config=model_config)

        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.667800784111023, places=TEST_EPS_PLACE)

    def test_lora_torch_compile(self):
        print("-----------------test_lora_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B",
            lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
            model_config=model_config,
        )
        video = generator.generate_video(
            prompt,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.22493965923786163, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
