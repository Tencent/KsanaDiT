import unittest

import torch
from pipeline_test_helper import PROMPTS, SEED, TEST_EPS_PLACE, TEST_FRAME_NUM, TEST_PORT, TEST_SIZE, TEST_STEPS

from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionBackend,
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSageSLAConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ksana.utils.distribute import get_gpu_count


class TestKsanaPipelineWanI2V(unittest.TestCase):

    def test_simple_i2v(self):
        print("-----------------test_simple_i2v-----------------")
        pipeline = KsanaPipeline.from_models("./Wan2.2-I2V-A14B", dist_config=KsanaDistributedConfig(port=TEST_PORT))
        videos = pipeline.generate(
            PROMPTS[0],
            img_path="./examples/images/input.png",
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
            ),
        )
        with self.subTest(msg="bs1 Shape Check"):
            # 576 is from image shape and target shape
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, 576, 576])
        mean0 = videos.cpu().abs().mean().item()

        videos = pipeline.generate(
            PROMPTS,
            img_path="./examples/images/start_image.png",
            end_img_path="./examples/images/end_image.png",
            sample_config=KsanaSampleConfig(steps=TEST_STEPS),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                rope_function="comfy",
                save_output=True,
            ),
        )
        mean1 = videos[0].cpu().abs().mean().item()
        mean2 = videos[1].cpu().abs().mean().item()
        with self.subTest(msg="bs 2 Shape Check"):
            self.assertEqual(list(videos.shape), [2, 3, TEST_FRAME_NUM, 576, 576])
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6250113248825073, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.47325804829597473, places=TEST_EPS_PLACE)

        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.4958144426345825, places=TEST_EPS_PLACE)

    def test_turbo_wan_i2v(self):
        print("-----------------test_turbo_wan_i2v-----------------")
        sage_sla_config = KsanaSageSLAConfig(
            dense_attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN), topk=0.1
        )

        model_config = KsanaModelConfig(attention_config=sage_sla_config, run_dtype=torch.bfloat16)

        sample_config = KsanaSampleConfig(steps=TEST_STEPS, cfg_scale=1.0, shift=5.0, solver=KsanaSolverType.EULER)

        high = "./TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-high-720P.pth"
        low = "./TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-low-720P.pth"
        text_dir = "./Wan2.2-I2V-A14B"
        vae_dir = "./Wan2.2-I2V-A14B"

        pipeline = KsanaPipeline.from_models(
            (high, low),
            text_checkpoint_dir=text_dir,
            vae_checkpoint_dir=vae_dir,
            model_config=model_config,
            dist_config=KsanaDistributedConfig(port=TEST_PORT),
        )

        videos = pipeline.generate(
            PROMPTS[0],
            img_path="./examples/images/input.png",
            sample_config=sample_config,
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
            ),
        )
        with self.subTest(msg="bs1 Shape Check"):
            self.assertEqual(list(videos.shape), [1, 3, TEST_FRAME_NUM, 576, 576])
        mean0 = videos.cpu().abs().mean().item()

        videos = pipeline.generate(
            PROMPTS,
            img_path="./examples/images/start_image.png",
            end_img_path="./examples/images/end_image.png",
            sample_config=sample_config,
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                rope_function="comfy",
                save_output=True,
            ),
        )
        mean1 = videos[0].cpu().abs().mean().item()
        mean2 = videos[1].cpu().abs().mean().item()
        places = TEST_EPS_PLACE if get_gpu_count() == 1 else 1

        with self.subTest(msg="bs 2 Shape Check"):
            self.assertEqual(list(videos.shape), [2, 3, TEST_FRAME_NUM, 576, 576])
        with self.subTest(msg="Mean 0 Check"):
            self.assertAlmostEqual(mean0, 0.6127950549125671, places=places)

        with self.subTest(msg="Mean 1 Check"):
            self.assertAlmostEqual(mean1, 0.44853246212005615, places=places)

        with self.subTest(msg="Mean 2 Check"):
            self.assertAlmostEqual(mean2, 0.45431801676750183, places=places)


if __name__ == "__main__":
    unittest.main()
