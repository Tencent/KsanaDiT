import os
import unittest

from nodes_test_helper import COMFY_MODEL_DIFFUSION_ROOT, TEST_MODELS, TEST_STEPS, run_load_and_generate

from ksana.config import KsanaLinearBackend
from ksana.utils.distribute import get_rank_id


class TestLinearForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, linear_backend):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            steps=TEST_STEPS,
            linear_backend=linear_backend,
        )
        self.assertEqual(load_output.model, expected_model_key)
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    def test_all_linear_backend(self):
        for model_name, img_shape, text_shape, expected_model_key in TEST_MODELS:
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
                    self.run_once(model_name, img_shape, text_shape, expected_model_key, linear_backend)


if __name__ == "__main__":
    unittest.main()
