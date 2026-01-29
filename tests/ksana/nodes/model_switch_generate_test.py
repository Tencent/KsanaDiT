import os
import unittest
from dataclasses import dataclass

from nodes_test_helper import (
    COMFY_MODEL_DIFFUSION_ROOT,
    IMG_SHAPE_I2V,
    IMG_SHAPE_T2I,
    IMG_SHAPE_T2V,
    QWEN_TEXT_SHAPE,
    TEST_GPUS_EPS_PLACE,
    TEST_ONE_GPU_EPS_PLACE,
    WAN_TEXT_SHAPE,
    run_load_and_generate,
)

from ksana.config import KsanaAttentionBackend, KsanaLinearBackend
from ksana.models.model_key import KsanaModelKey
from ksana.utils.distribute import get_gpu_count, get_rank_id

TEST_STEPS = 1


@dataclass
class KsanaNodesTestCase:
    model_names: list[str]
    image_latent_shape: list[int]
    attention_backends: KsanaAttentionBackend
    linear_backends: KsanaLinearBackend
    rope_function: str
    expect_model_key: KsanaModelKey
    expect__one_generator_output: float
    expect_gpus_generator_output: float


test_cases = [
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="comfy",
        expect_model_key=KsanaModelKey.Wan2_2_T2V_14B,
        expect__one_generator_output=0.76318359375,
        expect_gpus_generator_output=0.76318359375,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        ],
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.DEFAULT,
        rope_function="default",
        expect_model_key=KsanaModelKey.Wan2_2_I2V_14B,
        expect__one_generator_output=0.7939453125,
        expect_gpus_generator_output=0.7939453125,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", None],
        expect_model_key=KsanaModelKey.Wan2_2_T2V_14B,
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        expect__one_generator_output=0.77978515625,
        expect_gpus_generator_output=0.77978515625,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_t2v_high_noise_14B_fp16.safetensors",
            "wan2.2_t2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_key=KsanaModelKey.Wan2_2_T2V_14B,
        image_latent_shape=IMG_SHAPE_T2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="default",
        expect__one_generator_output=0.759765625,
        expect_gpus_generator_output=0.759765625,
    ),
    KsanaNodesTestCase(
        model_names=[
            "wan2.2_i2v_high_noise_14B_fp16.safetensors",
            "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        ],
        expect_model_key=KsanaModelKey.Wan2_2_I2V_14B,
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=KsanaAttentionBackend.SAGE_ATTN,
        linear_backends=KsanaLinearBackend.FP16_GEMM,
        rope_function="default",
        expect__one_generator_output=0.79052734375,
        expect_gpus_generator_output=0.79052734375,
    ),
    KsanaNodesTestCase(
        model_names=["wan2.2_i2v_high_noise_14B_fp16.safetensors", None],
        expect_model_key=KsanaModelKey.Wan2_2_I2V_14B,
        image_latent_shape=IMG_SHAPE_I2V,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM_DYNAMIC,
        rope_function="comfy",
        expect__one_generator_output=0.79052734375,
        expect_gpus_generator_output=0.79052734375,
    ),
    KsanaNodesTestCase(
        model_names="qwen_image_2512_fp8_e4m3fn.safetensors",
        expect_model_key=KsanaModelKey.QwenImage_T2I,
        image_latent_shape=IMG_SHAPE_T2I,
        attention_backends=KsanaAttentionBackend.FLASH_ATTN,
        linear_backends=KsanaLinearBackend.FP8_GEMM,
        rope_function="comfy",
        expect__one_generator_output=0.283203125,
        expect_gpus_generator_output=0.283203125,
    ),
]


class TestModelSwitchAndGenerate(unittest.TestCase):
    def test_base_and_swith_models(self):
        print("-----------------test_swith_models_and_generate-----------------")

        for test_case in test_cases:
            print(f"----------- test model_name: {test_case.model_names} -------------")
            if test_case.expect_model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
                high_noise_model_path = os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names[0])
                low_noise_model_path = (
                    os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names[1])
                    if test_case.model_names[1]
                    else None
                )

            else:
                high_noise_model_path = os.path.join(COMFY_MODEL_DIFFUSION_ROOT, test_case.model_names)
                low_noise_model_path = None
            if test_case.expect_model_key in [KsanaModelKey.QwenImage_T2I]:
                text_shape = QWEN_TEXT_SHAPE
            else:
                text_shape = WAN_TEXT_SHAPE

            load_output, generate_output = run_load_and_generate(
                high_noise_model_path,
                test_case.image_latent_shape,
                text_shape,
                TEST_STEPS,
                model_boundary=0.5,
                attn_backend=test_case.attention_backends,
                linear_backend=test_case.linear_backends,
                low_noise_model_path=low_noise_model_path,
                rope_function=test_case.rope_function,
                low_sample_guide_scale=3.0,
            )
            self.assertEqual(load_output.model, test_case.expect_model_key)
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
    def test_cache(self):
        pass


if __name__ == "__main__":
    unittest.main()
