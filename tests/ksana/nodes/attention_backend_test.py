import os
import unittest

import torch
from test_helper import COMFY_MODEL_ROOT, RUN_DTYPE, SEED, TEST_MODELS  # noqa # pylint: disable=unused-import

import ksana.nodes as nodes
from ksana import KsanaAttentionConfig, get_engine  # noqa # pylint: disable=unused-import
from ksana.utils.distribute import get_rank_id


class TestAttentionsForAllModels(unittest.TestCase):

    def run_once(self, model_name, image_latent_shape, text_shape, attn_backend):
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
            attention_config=KsanaAttentionConfig(backend=attn_backend),
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

    # TODO(TJ): need fix memory issue
    # def test_all_attention_backend(self):
    #     for model_name, img_shape, text_shape in TEST_MODELS:
    #         for attn_backend in KsanaAttentionBackend.get_supported_list():
    #             print(f"-----------------test {model_name} {attn_backend} -----------------")
    #             with self.subTest(msg=f"test {model_name} with {attn_backend}"):
    #                 self.run_once(model_name, img_shape, text_shape, attn_backend)


if __name__ == "__main__":
    unittest.main()
