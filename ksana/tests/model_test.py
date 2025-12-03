import unittest
from ksana import KsanaGenerator
from ksana.config import (
    KsanaModelConfig,
    KsanaTorchCompileConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaDistributedConfig,
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
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        self.assertAlmostEqual(mean0, 0.22557629644870758, places=TEST_EPS_PLACE)
        self.assertAlmostEqual(mean1, 0.24775567650794983, places=TEST_EPS_PLACE)

    def test_simple_gpus(self):
        print("-----------------test_simple_gpus-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=2))
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
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        self.assertAlmostEqual(mean0, 0.22557629644870758, places=TEST_EPS_PLACE)
        self.assertAlmostEqual(mean1, 0.24775567650794983, places=TEST_EPS_PLACE)

    def test_larger_seq(self):
        print("-----------------test_larger_seq-----------------")
        generator = KsanaGenerator.from_models("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompts[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=(1280, 720),
                frame_num=81,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.5367064476013184, places=TEST_EPS_PLACE)

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
                size=(720, 480),
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6427356600761414, places=TEST_EPS_PLACE)

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
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6555444598197937, places=TEST_EPS_PLACE)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        generator = KsanaGenerator.from_models(
            "./Wan2.2-T2V-A14B", lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"
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
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.2550005614757538, places=TEST_EPS_PLACE)

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
                return_frames=True,
                save_video=False,
            ),
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.22569085657596588, places=TEST_EPS_PLACE)

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
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.25498297810554504, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
