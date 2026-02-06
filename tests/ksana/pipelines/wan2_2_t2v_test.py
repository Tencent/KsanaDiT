# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    get_platform_config_or_skip,
)

from ksana import KsanaPipeline
from ksana.accelerator import platform
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
    MODEL_CONFIG = KsanaModelConfig(attention_config=KsanaAttentionConfig())

    def test_batch_size_per_prompt(self):
        print("-----------------test_batch_size_per_prompt-----------------")
        config = {
            "GPU": {
                "mean0": 0.666492760181427,
                "mean1": 0.6856070756912231,
                "mean2": 0.6659576892852783,
            },
            "NPU": {
                "mean0": 0.6717330813407898,
                "mean1": 0.6497427821159363,
                "mean2": 0.6722963452339172,
            },
        }
        expected_means = get_platform_config_or_skip(config, test_name="wan_t2v.test_batch_size_per_prompt")
        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            model_config=self.MODEL_CONFIG,
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
        )
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
        places = TEST_EPS_PLACE if platform.is_gpu() else 1
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, expected_means["mean0"], places=places)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, expected_means["mean1"], places=places)

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
            self.assertAlmostEqual(mean2, expected_means["mean2"], places=TEST_EPS_PLACE)

    def test_larger_seq_batch(self):
        print("-----------------test_larger_seq_batch-----------------")
        config = {
            "GPU": {
                "mean0": 0.518387496471405,
                "mean1": 0.2239505499601364,
            },
            "NPU": {
                "mean0": 0.43504077196121216,
                "mean1": 0.27056941390037537,
            },
        }
        expected_means = get_platform_config_or_skip(config, test_name="wan_t2v.test_larger_seq_batch")
        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            model_config=self.MODEL_CONFIG,
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
        )
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
        places = TEST_EPS_PLACE if platform.is_gpu() else 1
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, expected_means["mean0"], places=places)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, expected_means["mean1"], places=places)

    @unittest.skipIf(not platform.is_gpu(), "FP8 pipeline test runs only on GPU")
    def test_fp8(self):
        print("-----------------test_fp8-----------------")
        config = {"GPU": {"mean0": 0.66}}
        expected = get_platform_config_or_skip(config, test_name="wan_t2v.test_fp8")
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
        self.assertAlmostEqual(mean, expected["mean0"], places=1)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        config = {"GPU": {"mean0": 0.24302400648593903}, "NPU": {"mean0": 0.2562920153141022}}
        expected = get_platform_config_or_skip(config, test_name="wan_t2v.test_lora")
        pipeline = KsanaPipeline.from_models(
            "./Wan2.2-T2V-A14B",
            lora_config=KsanaLoraConfig("./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"),
            model_config=self.MODEL_CONFIG,
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
        self.assertAlmostEqual(mean, expected["mean0"], places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
