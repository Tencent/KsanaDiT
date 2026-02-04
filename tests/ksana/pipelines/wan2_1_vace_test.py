import unittest

import torch
from pipeline_test_helper import get_platform_config_or_skip

from ksana import KsanaPipeline
from ksana.config import (
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)

prompts = [
    "a cute anime girl with massive fennec ears and a big fluffy tail "
    "turning around and dancing and singing on stage like an idol",
]

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED = 10086
TEST_DTYPE = torch.float16
TEST_SIZE = (848, 480)
TEST_STEPS = 1
TEST_FRAME_NUM = 9
TEST_EPS_PLACE = 5


class TestKsanaPipelineWanVace(unittest.TestCase):
    def _assert_video_tensor_ok(self, video: torch.Tensor):
        self.assertIsInstance(video, torch.Tensor)
        self.assertEqual(video.ndim, 5)
        self.assertEqual(list(video.shape[:2]), [1, 3])
        self.assertEqual(list(video.shape[2:]), [TEST_FRAME_NUM, TEST_SIZE[1], TEST_SIZE[0]])
        self.assertTrue(torch.isfinite(video).all().item())

    def test_simple(self):
        print("-----------------wan2.1 vace test_simple-----------------")
        config = {
            "GPU": {"mean0": 0.80013597},
        }
        expected = get_platform_config_or_skip(config, test_name="wan2_1_vace.test_simple")
        pipeline = KsanaPipeline.from_models("./Wan2.1-VACE-14B")
        video = pipeline.generate(
            prompts[0],
            prompt_negative=NEGATIVE_PROMPT,
            sample_config=KsanaSampleConfig(
                steps=TEST_STEPS,
                cfg_scale=5.0,
                shift=5.0,
                solver=KsanaSolverType.UNI_PC,
            ),
            runtime_config=KsanaRuntimeConfig(
                seed=SEED,
                size=TEST_SIZE,
                frame_num=TEST_FRAME_NUM,
                return_frames=True,
                save_output=True,
            ),
        )
        self._assert_video_tensor_ok(video)
        mean = video.detach().float().abs().mean().item()
        self.assertAlmostEqual(mean, expected["mean0"], places=TEST_EPS_PLACE)
        self.assertAlmostEqual(mean, 0.80013597, places=TEST_EPS_PLACE)


if __name__ == "__main__":
    unittest.main()
