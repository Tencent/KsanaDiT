import os
import unittest

from test_helper import (
    COMFY_MODEL_DIFFUSION_ROOT,
    TEST_STEPS,
    run_load_and_generate,
)

from ksana.utils.distribute import get_rank_id


class TestAttentionsForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, attn_backend):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            TEST_STEPS,
            attn_backend=attn_backend,
        )
        self.assertEqual(load_output.model, expected_model_key)
        generate_output = generate_output.samples
        if get_rank_id() == 0:
            # only return tensor on rank 0
            self.assertIsNotNone(generate_output)
        else:
            self.assertIsNone(generate_output)

    # TODO(TJ): need fix memory issue
    # def test_all_attention_backend(self):
    #     for model_name, img_shape, text_shape, expected_model_key in TEST_MODELS:
    #         for attn_backend in KsanaAttentionBackend.get_supported_list():
    #             print(f"-----------------test {model_name} {attn_backend} -----------------")
    #             with self.subTest(msg=f"test {model_name} with {attn_backend}"):
    #                 self.run_once(model_name, img_shape, text_shape, expected_model_key, attn_backend)


if __name__ == "__main__":
    unittest.main()
