import os
import unittest

import torch
from test_helper import COMFY_MODEL_ROOT, RUN_DTYPE, SEED, TEST_MODELS

import ksana.nodes as nodes
from ksana import get_engine
from ksana.config import KsanaLinearBackend
from ksana.utils.distribute import get_rank_id


class TestLinearForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, linear_backend):
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
            high_noise_model_path=os.path.join(COMFY_MODEL_ROOT, model_name),
            linear_backend=linear_backend,
        )

        image_latent = torch.zeros(*image_latent_shape, dtype=RUN_DTYPE, device="cpu")
        generate_output = nodes.generate(
            output,
            positive=[[positive_text_embeddings]],
            negative=[[negtive_text_embeddings]],
            latent_image=nodes.KsanaNodeVAEEncodeOutput(samples=image_latent),
            steps=1,
            seed=SEED,
        )
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def test_base_and_swith_models(self):
        ksana_engine = get_engine()
        ksana_engine.clear_models()
        for model_name, img_shape, text_shape in TEST_MODELS:
            for linear_backend in KsanaLinearBackend.get_supported_list():
                if KsanaLinearBackend(linear_backend) == KsanaLinearBackend.FP8_GEMM and "fp8" not in model_name:
                    # Note: fp8_gemm only can used in fp8
                    print(f"-----------------skip test {model_name} {linear_backend} -----------------")
                    continue
                # if (
                #     KsanaLinearBackend(linear_backend) == KsanaLinearBackend.FP8_GEMM_DYNAMIC
                #     and "fp16" not in model_name
                # ):
                #     # Note: fp8_gemm_dynamic only need used in fp16 but still can work in fp8
                #     print(f"-----------------skip test {model_name} {linear_backend} -----------------")
                #     continue
                print(f"-----------------test {model_name} {linear_backend} -----------------")
                with self.subTest(msg=f"test {model_name} with {linear_backend}"):
                    self.run_once(model_name, img_shape, text_shape, linear_backend)


if __name__ == "__main__":
    unittest.main()
