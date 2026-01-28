import unittest

import torch
from pipeline_test_helper import (
    PROMPTS,
    RADIAL_ATTN_TEST_FRAME_NUM,
    RADIAL_ATTN_TEST_SIZE,
    SEED,
    TEST_EPS_PLACE,
    TEST_FRAME_NUM,
    TEST_PORT,
    TEST_SIZE,
    TEST_STEPS,
)

from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionBackend,
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaLinearBackend,
    KsanaLoraConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaTorchCompileConfig,
)


class TestKsanaPipelineWanT2V(unittest.TestCase):

    def test_batch_size_per_prompt(self):
        print("-----------------test_batch_size_per_prompt-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B", dist_config=KsanaDistributedConfig(port=TEST_PORT))
        batch_size_per_prompts = [2, 3]
        videos = pipeline.generate(
            PROMPTS,
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
                batch_size_per_prompts=batch_size_per_prompts,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(
                list(videos.shape), [sum(batch_size_per_prompts), 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]]
            )
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()

        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.666492760181427, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.6856070756912231, places=TEST_EPS_PLACE)

        videos = pipeline.generate(
            PROMPTS[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                rope_function="comfy",
                return_frames=True,
                save_output=True,
            ),
        )
        with self.subTest(msg="bs1 Shape Check"):
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        mean2 = videos.cpu().abs().mean().item()
        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.6659576892852783, places=TEST_EPS_PLACE)

    def test_larger_seq_batch(self):
        print("-----------------test_larger_seq_batch-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-T2V-A14B", dist_config=KsanaDistributedConfig(port=TEST_PORT))
        videos = pipeline.generate(
            PROMPTS,
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
            self.assertEqual(list(videos.shape), [len(PROMPTS), 3, 81, 720, 1280])
        mean0 = videos[0].cpu().abs().mean().item()
        mean1 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.518387496471405, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.2239505499601364, places=TEST_EPS_PLACE)

    def test_fp8(self):
        print("-----------------test_fp8-----------------")
        low_noise_model_path = "./comfy_models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
        high_noise_model_path = "./comfy_models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
        text_dir = "./Wan2.2-T2V-A14B"
        vae_dir = "./Wan2.2-T2V-A14B"
        model_config = KsanaModelConfig(
            run_dtype=torch.float16,
            attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
            linear_backend=KsanaLinearBackend.FP8_GEMM,
            torch_compile_config=KsanaTorchCompileConfig(),
        )

        pipeline = KsanaPipeline.from_models(
            (high_noise_model_path, low_noise_model_path),  # high go first
            text_checkpoint_dir=text_dir,
            vae_checkpoint_dir=vae_dir,
            model_config=model_config,
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
        )
        video = pipeline.generate(
            PROMPTS[0],
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
        self.assertAlmostEqual(mean, 0.66, places=1)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            lora_config=KsanaLoraConfig("./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"),
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
        )

        video = pipeline.generate(
            PROMPTS[0],
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=RADIAL_ATTN_TEST_SIZE,
                frame_num=RADIAL_ATTN_TEST_FRAME_NUM,
                return_frames=True,
                save_output=False,
            ),
        )
        with self.subTest(msg="Shape Check"):
            self.assertEqual(
                list(video.shape),
                [1, 3, RADIAL_ATTN_TEST_FRAME_NUM, RADIAL_ATTN_TEST_SIZE[1], RADIAL_ATTN_TEST_SIZE[0]],
            )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.24302400648593903, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
