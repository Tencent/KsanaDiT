import os
import unittest

from nodes_test_helper import COMFY_MODEL_DIFFUSION_ROOT, TEST_MODELS, run_load_and_generate

from ksana.utils.distribute import get_rank_id

TEST_STEPS = 2

PROMPT_SIZE = [1, 3]
BATCH_SIZE_PER_PROMPT = [1, 2]


class TestBatchingForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, expected_model_key, batch_size_per_prompt):
        load_output, generate_output = run_load_and_generate(
            os.path.join(COMFY_MODEL_DIFFUSION_ROOT, model_name),
            image_latent_shape,
            text_shape,
            TEST_STEPS,
            batch_size_per_prompt=batch_size_per_prompt,
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
            for prompt_size in PROMPT_SIZE:
                text_shape = text_shape.copy()
                text_shape[0] = prompt_size
                for batch_size_per_prompt in BATCH_SIZE_PER_PROMPT:
                    print(
                        f"-----------------test {model_name} prompt_size:{prompt_size} "
                        f"batch_size_per_prompt:{batch_size_per_prompt} -----------------"
                    )
                    with self.subTest(
                        msg=f"test {model_name} with prompt_size:{prompt_size} "
                        f"batch_size_per_prompt:{batch_size_per_prompt}"
                    ):
                        self.run_once(model_name, img_shape, text_shape, expected_model_key, batch_size_per_prompt)


if __name__ == "__main__":
    unittest.main()
