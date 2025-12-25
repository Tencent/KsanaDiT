import unittest
from ksana import KsanaGenerator
from ksana.config import (
    KsanaModelConfig,
    KsanaTorchCompileConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)
import torch

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


class TestKsana(unittest.TestCase):

    def test_simple(self):
        print("-----------------test_simple-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        videos = generator.generate_video(
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
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6556181311607361, places=5)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.4420677423477173, places=5)

    def test_larger_seq_batch(self):
        print("-----------------test_larger_seq_batch-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        videos = generator.generate_video(
            prompts,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=(1280, 720),
                frame_num=81,
                return_frames=True,
                save_video=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(videos.shape), [len(prompts), 3, 81, 720, 1280])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.5367065072059631, places=5)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.24088804423809052, places=5)

    def test_fp8(self):
        print("-----------------test_fp8-----------------")
        low_noise_model_path = "./comfy_models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        high_noise_model_path = "./comfy_models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        text_dir = "./Wan2.2-T2V-A14B"
        vae_dir = "./Wan2.2-T2V-A14B"
        model_config = KsanaModelConfig(
            run_dtype=torch.float16,
            attn_backend="sage_attention",
            linear_backend="fp8_gemm",
            torch_compile_config=KsanaTorchCompileConfig(),
        )

        generator = KsanaGenerator.from_models(
            (high_noise_model_path, low_noise_model_path),  # high go first
            text_checkpoint_dir=text_dir,
            vae_checkpoint_dir=vae_dir,
            model_config=model_config,
        )
        video = generator.generate_video(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.658674955368042, places=TEST_EPS_PLACE)

    def test_cache(self):
        # TODO: step 1 can not test cache, real test cache logical,
        print("-----------------test_cache-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompts[0],
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
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6556134819984436, places=TEST_EPS_PLACE)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B", lora="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"
        )
        video = generator.generate_video(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.255082368850708, places=TEST_EPS_PLACE)

    def test_torch_compile(self):
        print("-----------------test_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B", model_config=model_config)

        video = generator.generate_video(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                rope_function="comfy",
                return_frames=True,
                save_video=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6557297706604004, places=TEST_EPS_PLACE)

    def test_lora_torch_compile(self):
        print("-----------------test_lora_torch_compile-----------------")
        model_config = KsanaModelConfig(
            run_dtype=TEST_DTYPE,
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B",
            lora="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
            model_config=model_config,
        )
        video = generator.generate_video(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(list(video.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.25497761368751526, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
