import os
import unittest
from dataclasses import dataclass

import torch

import ksana.nodes as nodes
from ksana import KsanaAttentionConfig, get_engine
from ksana.config import KsanaAttentionBackend
from ksana.models.model_key import KsanaModelKey
from ksana.operations import KsanaLinearBackend
from ksana.utils.distribute import get_gpu_count, get_rank_id

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models/diffusion_models"
SEED = 321
RUN_DTYPE = torch.float16
TEST_ONE_GPU_EPS_PLACE = 6
TEST_GPUS_EPS_PLACE = 6

TARGET_T2V_IMG_SHAPE = [1, 16, 16, 32, 32]
TARGET_I2V_IMG_SHAPE = [1, 20, 16, 32, 32]


@dataclass
class KsanaNodesTestCase:
    model_names: list[str]
    image_latent_shape: list[int]
    attention_backends: KsanaAttentionBackend
    linear_backends: KsanaLinearBackend
    rope_function: str
    expect_model_keys: list[KsanaModelKey]
    expect__one_generator_output: float
    expect_gpus_generator_output: float


test_cases = [
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=TARGET_T2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="comfy",
        expect_model_keys=[KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
        expect__one_generator_output=0.75537109375,
        expect_gpus_generator_output=0.755859375,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=TARGET_I2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="default",
        expect_model_keys=[KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
        expect__one_generator_output=0.77734375,
        expect_gpus_generator_output=0.77734375,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", None],
        expect_model_keys=[KsanaModelKey.Wan2_2_T2V_14B_HIGH],
        image_latent_shape=TARGET_T2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        expect__one_generator_output=0.7744140625,
        expect_gpus_generator_output=0.7744140625,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp16.safetensors",
            "wan2.2_t2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_keys=[KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
        image_latent_shape=TARGET_T2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="default",
        expect__one_generator_output=0.75439453125,
        expect_gpus_generator_output=0.75439453125,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp16.safetensors",
            "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_keys=[KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
        image_latent_shape=TARGET_I2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.FP16_GEMM,
        rope_function="default",
        expect__one_generator_output=0.7763671875,
        expect_gpus_generator_output=0.7763671875,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_i2v_high_noise_14B_fp16.safetensors", None],
        expect_model_keys=[KsanaModelKey.Wan2_2_I2V_14B_HIGH],
        image_latent_shape=TARGET_I2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="comfy",
        expect__one_generator_output=0.7763671875,
        expect_gpus_generator_output=0.7763671875,
    ),
]


class TestNodes(unittest.TestCase):
    def test_base_and_swith_models(self):
        print("-----------------test_base_and_swith_models-----------------")
        ksana_engine = get_engine()
        ksana_engine.clear_models()

        seed_g = torch.Generator(device="cpu")
        seed_g.manual_seed(SEED)
        text_shape = [1, 512, 4096]
        positive_text_embeddings = torch.randn(
            *text_shape,
            dtype=RUN_DTYPE,
            device="cpu",
            generator=seed_g,
        )
        negtive_text_embeddings = torch.randn(
            *text_shape,
            dtype=RUN_DTYPE,
            device="cpu",
            generator=seed_g,
        )

        for test_case in test_cases:
            print(f"----------- test model_name: {test_case.model_names} -------------")
            high_noise_model_path = os.path.join(COMFY_MODEL_ROOT, test_case.model_names[0])
            low_noise_model_path = (
                os.path.join(COMFY_MODEL_ROOT, test_case.model_names[1]) if test_case.model_names[1] else None
            )
            output = nodes.KsanaNodeModelLoader.load(
                high_noise_model_path=high_noise_model_path,
                low_noise_model_path=low_noise_model_path,
                attention_config=KsanaAttentionConfig(
                    backend=test_case.attention_backends,
                ),
                linear_backend=test_case.linear_backends,
                model_boundary=0.5,
            )
            self.assertEqual(output.model, test_case.expect_model_keys)

            image_latent = torch.zeros(
                *test_case.image_latent_shape,
                dtype=RUN_DTYPE,
                device="cpu",
            )
            generate_output = nodes.generate(
                output,
                positive=[[positive_text_embeddings]],
                negative=[[negtive_text_embeddings]],
                latent_image=nodes.KsanaNodeVAEEncodeOutput(samples=image_latent),
                steps=1,
                seed=SEED,
                rope_function=test_case.rope_function,
                low_sample_guide_scale=3.0,
            )
            generate_output = generate_output.samples
            if get_rank_id() == 0:
                # only return tensor on rank 0
                self.assertIsNotNone(generate_output)
            else:
                self.assertIsNone(generate_output)
                continue

            with self.subTest(msg=f"KsanaNodesTestCase {test_case} generate shape Check"):
                target_latent_shape = test_case.image_latent_shape.copy()
                target_latent_shape[1] = 16  # always 16
                self.assertEqual(list(generate_output.shape), target_latent_shape)
            mean = generate_output.cpu().abs().mean().item()
            with self.subTest(msg=f"KsanaNodesTestCase {test_case} generate output mean {mean} check"):
                if get_gpu_count() == 1:
                    self.assertAlmostEqual(mean, test_case.expect__one_generator_output, places=TEST_ONE_GPU_EPS_PLACE)
                else:
                    self.assertAlmostEqual(mean, test_case.expect_gpus_generator_output, places=TEST_GPUS_EPS_PLACE)

    # TODO: for all models, only one high, load once, and test belows for
    def test_attention(self):
        pass

    def test_linear_backend(self):
        pass

    def test_cache(self):
        pass

    def test_lora(self):
        pass


if __name__ == "__main__":
    unittest.main()
