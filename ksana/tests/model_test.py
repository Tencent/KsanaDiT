import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import unittest
from ksana import KsanaGenerator
from ksana import KsanaTorchCompileConfig
import torch

prompt = (
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"
)
seed = 123
# TODO: add test dtype
dtype = torch.float16
eps_place = 7
test_steps = 1


class TestKsana(unittest.TestCase):

    def test_simple(self):
        print("-----------------test_simple-----------------")
        generator = KsanaGenerator.from_pretrained("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompt,
            steps=test_steps,
            size=(720, 480),
            frame_num=17,
            seed=seed,
            return_frames=True,
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.34484848380088806, places=eps_place)

    def test_cache(self):
        # TODO: step 1 can not test cache, real test cache logical,
        print("-----------------test_cache-----------------")
        generator = KsanaGenerator.from_pretrained("./Wan2.2-T2V-A14B")
        video = generator.generate_video(
            prompt,
            steps=test_steps,
            size=(720, 480),
            frame_num=9,
            seed=seed,
            cache_method="DCache",
            return_frames=True,
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.666985273361206, places=eps_place)

    def test_lora(self):
        print("-----------------test_lora-----------------")
        generator = KsanaGenerator.from_pretrained(
            "./Wan2.2-T2V-A14B", lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1"
        )
        video = generator.generate_video(
            prompt,
            steps=test_steps,
            size=(720, 480),
            frame_num=9,
            seed=seed,
            return_frames=True,
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.29763856530189514, places=eps_place)

    def test_torch_compile(self):
        print("-----------------test_torch_compile-----------------")
        generator = KsanaGenerator.from_pretrained("./Wan2.2-T2V-A14B", torch_compile_config=KsanaTorchCompileConfig())

        video = generator.generate_video(
            prompt,
            steps=test_steps,
            size=(720, 480),
            frame_num=9,
            seed=seed,
            return_frames=True,
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.6693246960639954, places=eps_place)

    def test_lora_torch_compile(self):
        print("-----------------test_lora_torch_compile-----------------")
        generator = KsanaGenerator.from_pretrained(
            "./Wan2.2-T2V-A14B",
            lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
            torch_compile_config=KsanaTorchCompileConfig(),
        )
        video = generator.generate_video(
            prompt,
            steps=test_steps,
            size=(720, 480),
            frame_num=9,
            seed=seed,
            return_frames=True,
        )
        mean = video.cpu().abs().mean().item()
        self.assertAlmostEqual(mean, 0.2974734306335449, places=eps_place)


if __name__ == "__main__":
    unittest.main()
