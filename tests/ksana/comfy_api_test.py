import os
import unittest

import ksana.comfy as ksana_comfy
from ksana.models.model_key import KsanaModelKey

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models/diffusion_models"


class TestComfyApi(unittest.TestCase):
    def test_model_loader(self):
        model_names = [
            ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
            ["wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", None],
        ]
        expect_keys = [
            [KsanaModelKey.Wan2_2_T2V_14B_HIGH, KsanaModelKey.Wan2_2_T2V_14B_LOW],
            [KsanaModelKey.Wan2_2_I2V_14B_HIGH],
        ]
        for model_name, expect_key in zip(model_names, expect_keys):
            high_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[0])
            low_noise_model_path = os.path.join(COMFY_MODEL_ROOT, model_name[1]) if model_name[1] else None
            output = ksana_comfy.KsanaComfyModelLoader.load(
                high_noise_model_path=high_noise_model_path,
                low_noise_model_path=low_noise_model_path,
            )
            self.assertEqual(output.model, expect_key)


if __name__ == "__main__":
    unittest.main()
