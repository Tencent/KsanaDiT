import os
import unittest
from dataclasses import dataclass

import torch

import ksana.nodes as nodes
from ksana import KsanaAttentionConfig, get_engine
from ksana.config import KsanaAttentionBackend
from ksana.models.model_key import KsanaModelKey
from ksana.operations import KsanaLinearBackend

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models/diffusion_models"
SEED = 321
RUN_DTYPE = torch.float16
TEST_EPS_PLACE = 7

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
    expect_generator_outputs: list[float]


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
        expect_generator_outputs=0.78076171875,
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
        expect_generator_outputs=0.79736328125,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", None],
        expect_model_keys=[KsanaModelKey.Wan2_2_T2V_14B_HIGH],
        image_latent_shape=TARGET_T2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        expect_generator_outputs=0.80517578125,
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
        expect_generator_outputs=0.77880859375,
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
        expect_generator_outputs=0.7939453125,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_i2v_high_noise_14B_fp16.safetensors", None],
        expect_model_keys=[KsanaModelKey.Wan2_2_I2V_14B_HIGH],
        image_latent_shape=TARGET_I2V_IMG_SHAPE,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="comfy",
        expect_generator_outputs=0.7939453125,
    ),
]


# TODO:(TJ): nodes should have both single and gpus
class TestNodes(unittest.TestCase):
    def test_model_loader(self):
        print("-----------------test_model_loader-----------------")
        # TODO(TJ): add engine gpus
        ksana_engine = get_engine()
        ksana_engine.clear_models()

        seed_g = torch.Generator(device="cuda")
        seed_g.manual_seed(SEED)
        text_shape = [1, 512, 4096]
        positive_text_embeddings = torch.randn(
            *text_shape,
            dtype=RUN_DTYPE,
            device="cuda",
            generator=seed_g,
        )
        negtive_text_embeddings = torch.randn(
            *text_shape,
            dtype=RUN_DTYPE,
            device="cuda",
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
                device="cuda",
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
            with self.subTest(msg="generate Shape Check"):
                target_latent_shape = test_case.image_latent_shape.copy()
                target_latent_shape[1] = 16  # always 16
                self.assertEqual(list(generate_output.shape), target_latent_shape)

            mean = generate_output.cpu().abs().mean().item()
            print(f"KsanaNodesTestCase:{test_case} output mean: {mean}")
            with self.subTest(msg="generate output Mean Check"):
                self.assertAlmostEqual(mean, test_case.expect_generator_outputs, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
