import os
import unittest

import torch
from test_helper import COMFY_MODEL_ROOT, RUN_DTYPE, SEED, TARGET_I2V_IMG_SHAPE, TEST_STEPS, WAN_TEXT_SHAPE

import ksana.nodes as nodes
from ksana.utils.distribute import get_rank_id

TEST_LORAS = [
    (os.path.join(COMFY_MODEL_ROOT, "loras", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise.safetensors"), 1),
    [
        (
            os.path.join(
                COMFY_MODEL_ROOT, "loras", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_high_noise.safetensors"
            ),
            0.3,
        ),
        (
            os.path.join(COMFY_MODEL_ROOT, "loras", "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1_low_noise.safetensors"),
            0.7,
        ),
    ],
]

TEST_MODELS = [
    ("wan2.2_i2v_high_noise_14B_fp16.safetensors", TARGET_I2V_IMG_SHAPE, WAN_TEXT_SHAPE, TEST_LORAS),
]


class TestLinearForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, lora_config):
        seed_g = torch.Generator(device="cpu")
        seed_g.manual_seed(SEED)
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

        output = nodes.KsanaNodeModelLoader.load(
            high_noise_model_path=os.path.join(COMFY_MODEL_ROOT, "diffusion_models", model_name),
            lora=nodes.build_list_of_lora_config(lora_config),
        )

        image_latent = torch.zeros(*image_latent_shape, dtype=RUN_DTYPE, device="cpu")
        generate_output = nodes.generate(
            output,
            positive=[[positive_text_embeddings]],
            negative=[[negtive_text_embeddings]],
            latent_image=nodes.KsanaNodeVAEEncodeOutput(samples=image_latent),
            steps=TEST_STEPS,
            seed=SEED,
        )
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def test_lora(self):
        for model_name, img_shape, text_shape, loras in TEST_MODELS:
            for lora_config in loras:
                print(f"-----------------test {model_name} {lora_config} -----------------")
                with self.subTest(msg=f"test {model_name} with {lora_config}"):
                    self.run_once(model_name, img_shape, text_shape, lora_config)


if __name__ == "__main__":
    unittest.main()
