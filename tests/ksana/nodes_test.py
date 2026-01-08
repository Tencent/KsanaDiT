import os
import unittest

import torch

import ksana.nodes as nodes
from ksana import KsanaAttentionConfig, get_engine
from ksana.config import KsanaAttentionBackend
from ksana.models.model_key import KsanaModelKey
from ksana.operations import KsanaLinearBackend

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models/diffusion_models"
SEED = 321
RUN_DTYPE = torch.float16

TARGET_T2V_IMG_SHAPE = [1, 16, 16, 32, 32]
TARGET_I2V_IMG_SHAPE = [1, 20, 16, 32, 32]


# TODO:(TJ): nodes should have both single and gpus
class TestNodes(unittest.TestCase):
    def test_model_loader(self):
        print("-----------------test_model_loader-----------------")
        # TODO(TJ): add engine gpus
        ksana_engine = get_engine()
        ksana_engine.clear_models()

        model_names = [
            ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
            ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"],
            ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", None],
            ["wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan2.2_t2v_low_noise_14B_fp16.safetensors"],
            # ["wan2.2_i2v_high_noise_14B_fp16.safetensors", "wan2.2_i2v_low_noise_14B_fp16.safetensors"],
            # ["wan2.2_i2v_high_noise_14B_fp16.safetensors", None],
        ]
        image_latent_shapes = [
            TARGET_T2V_IMG_SHAPE,
            TARGET_I2V_IMG_SHAPE,
            TARGET_T2V_IMG_SHAPE,
            TARGET_T2V_IMG_SHAPE,
            # TARGET_I2V_IMG_SHAPE,
            # TARGET_I2V_IMG_SHAPE,
        ]
        expect_model_keys = [
            [KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
            [KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
            [KsanaModelKey.Wan2_2_T2V_14B_HIGH],
            [KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
            # [KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
            # [KsanaModelKey.Wan2_2_I2V_14B_HIGH],
        ]

        attention_backends = [
            KsanaAttentionBackend.SAGE_ATTN,
            KsanaAttentionBackend.SAGE_ATTN,
            KsanaAttentionBackend.FLASH_ATTN,
            KsanaAttentionBackend.SAGE_ATTN,
            # KsanaAttentionBackend.FLASH_ATTN,
            # KsanaAttentionBackend.SAGE_ATTN,
        ]
        linear_backends = [
            KsanaLinearBackend.DEFAULT,
            KsanaLinearBackend.FP8_GEMM_DYNAMIC,
            KsanaLinearBackend.FP8_GEMM,
            KsanaLinearBackend.DEFAULT,
            # KsanaLinearBackend.FP8_GEMM_DYNAMIC,
            # KsanaLinearBackend.DEFAULT,
        ]

        rope_functions = [
            "comfy",
            "default",
            "comfy",
            "default",
            # "default",
            # "comfy",
        ]

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

        for model_name, expect_key, image_latent_shape, attention_backend, linear_backend, rope_function in zip(
            model_names, expect_model_keys, image_latent_shapes, attention_backends, linear_backends, rope_functions
        ):
            print(f"----------- test model_name: {model_name} -------------")
            high_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[0])
            low_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[1]) if model_name[1] else None
            output = nodes.KsanaNodeModelLoader.load(
                high_noise_model_path=high_noise_model_path,
                low_noise_model_path=low_noise_model_path,
                attention_config=KsanaAttentionConfig(
                    backend=attention_backend,
                ),
                linear_backend=linear_backend,
                model_boundary=0.5,
            )
            self.assertEqual(output.model, expect_key)

            image_latent = torch.zeros(
                *image_latent_shape,
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
                rope_function=rope_function,
                low_sample_guide_scale=3.0,
            )
            generate_output = generate_output.samples
            print(f"----------- generate_output.shape: {generate_output.shape}")
            with self.subTest(msg="bs1 Shape Check"):
                target_latent_shape = image_latent_shape
                target_latent_shape[1] = 16  # always 16
                self.assertEqual(list(generate_output.shape), target_latent_shape)

            mean = generate_output.cpu().abs().mean().item()
            print(f"----------- mean: {mean}")


if __name__ == "__main__":
    unittest.main()
