import os
import unittest

from nodes_test_helper import (
    COMFY_MODEL_DIFFUSION_ROOT,
    COMFY_MODEL_ROOT,
    IMG_SHAPE_I2V,
    TEST_STEPS,
    WAN_TEXT_SHAPE,
    run_load_and_generate,
)

from ksana.models import KsanaModelKey
from ksana.utils.distribute import get_rank_id

LORA_ROOT_PATH = os.path.join(COMFY_MODEL_ROOT, "loras")
TEST_LORAS = [
    (os.path.join(LORA_ROOT_PATH, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise.safetensors"), 1),
    [
        (
            os.path.join(LORA_ROOT_PATH, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise.safetensors"),
            0.3,
        ),
        (
            os.path.join(LORA_ROOT_PATH, "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_low_noise.safetensors"),
            0.7,
        ),
    ],
]

TEST_MODELS = [
    (
        "wan2.2_i2v_high_noise_14B_fp16.safetensors",
        IMG_SHAPE_I2V,
        WAN_TEXT_SHAPE,
        KsanaModelKey.Wan2_2_I2V_14B,
        TEST_LORAS,
    ),
]


class TestLorasForModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, lora_config):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            TEST_STEPS,
            lora_config=lora_config,
        )
        self.assertEqual(load_output.model, expected_model_key)
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def test_lora(self):
        for model_name, img_shape, text_shape, expected_model_key, loras in TEST_MODELS:
            for lora_config in loras:
                print(f"-----------------test {model_name} {lora_config} -----------------")
                with self.subTest(msg=f"test {model_name} with {lora_config}"):
                    self.run_once(model_name, img_shape, text_shape, expected_model_key, lora_config)


if __name__ == "__main__":
    unittest.main()
