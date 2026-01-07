import os
import unittest

import ksana.nodes as nodes
from ksana import get_generator
from ksana.models.model_key import KsanaModelKey

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models/diffusion_models"


# TODO:(TJ): nodes should have both single and gpus
class TestNodes(unittest.TestCase):
    def test_model_loader(self):
        print("-----------------test_model_loader-----------------")
        # TODO(TJ): add generator gpus
        ksana_generator = get_generator()
        ksana_generator.clear_models()

        model_names = [
            ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
            ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", None],
            ["wan2.2_i2v_high_noise_14B_fp16.safetensors", "wan2.2_i2v_low_noise_14B_fp16.safetensors"],
        ]
        expect_keys = [
            [KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
            [KsanaModelKey.Wan2_2_I2V_14B_HIGH],
            [KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
        ]
        for model_name, expect_key in zip(model_names, expect_keys):
            high_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[0])
            low_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[1]) if model_name[1] else None
            output = nodes.KsanaNodeModelLoader.load(
                high_noise_model_path=high_noise_model_path,
                low_noise_model_path=low_noise_model_path,
            )
            self.assertEqual(output.model, expect_key)

    # def test_attn_linear(self):
    #     model_names = [
    #         ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
    #         ["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"],
    #         ["wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan2.2_t2v_low_noise_14B_fp16.safetensors"],
    #         ["wan2.2_i2v_high_noise_14B_fp16.safetensors", "wan2.2_i2v_low_noise_14B_fp16.safetensors"],

    #         ["wan2.2_t2v_high_noise_14B_fp16_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
    #         ["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors", None],
    #     ]
    #     expect_keys = [
    #         [KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
    #         [KsanaModelKey.Wan2_2_I2V_14B_HIGH],
    #         [KsanaModelKey.Wan2_2_I2V_14B_HIGH, KsanaModelKey.Wan2_2_I2V_14B_LOW],
    #     ]
    #     text_shape = [1, 512, 4096]
    #     seed = (
    #         runtime_config.seed
    #         if runtime_config.seed is not None and runtime_config.seed >= 0
    #         else random.randint(0, sys.maxsize)
    #     )
    #     single_noise = torch.randn(
    #             vae_z_dim,
    #             target_shape[2],
    #             target_shape[3],
    #             target_shape[4],
    #             dtype=torch.float32,
    #             device=device,
    #             generator=seed_g,
    #         ).to(dtype)

    #     torch.random.manual_seed(321)
    #     for model_name, expect_key in zip(model_names, expect_keys):
    #         high_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[0])
    #         low_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[1]) if model_name[1] else None
    #         output = nodes.KsanaNodeModelLoader.load(
    #             high_noise_model_path=high_noise_model_path,
    #             low_noise_model_path=low_noise_model_path,
    #         )
    #         self.assertEqual(output.model, expect_key)


if __name__ == "__main__":
    unittest.main()
