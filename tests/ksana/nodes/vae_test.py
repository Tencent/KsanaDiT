# Copyright 2026 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import unittest

import ksana.nodes as nodes
import torch
import torch.distributed as dist
from ksana import get_engine
from ksana.utils.distribute import get_rank_id
from nodes_test_helper import (
    COMFY_MODEL_ROOT,
    CURRENT_PLATFORM,
    SEED,
)

VAE_MODEL_PATH = os.path.join(COMFY_MODEL_ROOT, "VAE", "Wan2.1_VAE.pth")

IN_DIMS = [1, 3, 81, 640, 640]
LATENT_CHANNELS = 16
LATENT_SHAPE = [1, 16, 21, 80, 80]

TEST_EPS_PLACE = 1


def _make_generator():
    g = torch.Generator(device="cpu")
    g.manual_seed(SEED)
    return g


def _ensure_tensor(x):
    return torch.stack(x, dim=0) if isinstance(x, list) else x


def _encode_decode(vae_model, input_tensor, rank):
    latents = _ensure_tensor(vae_model.encode(input_tensor))
    frames = _ensure_tensor(vae_model.decode(latents))
    if rank == 0:
        print(f"latents - {latents.shape} {latents.min():.4f}~{latents.max():.4f}")
        print(f"  [checksum] mean={latents.float().mean().item():.8f} sum={latents.float().sum().item():.4f}")
        print(f"frames  - {frames.shape} {frames.min():.4f}~{frames.max():.4f}")
        print(f"  [checksum] mean={frames.float().mean().item():.8f} sum={frames.float().sum().item():.4f}")
    return latents, frames


def _benchmark(fn, device, warmup=3, repeats=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    t0 = time.time()
    for _ in range(repeats):
        fn()
        torch.cuda.synchronize(device)
    return (time.time() - t0) / repeats * 1000


class TestWan21VAENodeParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.bfloat16
        # Note: 必须先调用 load()，其内部 get_engine() → KsanaExecutor(device_id=local_rank_id)
        # 会执行 torch.cuda.set_device()，之后 current_device() 才能返回正确的设备 ID
        cls.vae_model_key = nodes.KsanaNodeVAELoader.load(vae_path=VAE_MODEL_PATH)
        cls.device = torch.cuda.current_device()
        cls.rank = get_rank_id()
        cls.world_size = dist.get_world_size() if dist.is_initialized() else 1

        ksana_engine = get_engine()
        executor = ksana_engine.executors if not isinstance(ksana_engine.executors, list) else ksana_engine.executors[0]
        cls.vae = executor.model_pool.get_model(cls.vae_model_key)
        cls.vae.model.to(cls.device)

    def test_encode_decode_random(self):
        """Test VAE encode/decode roundtrip with random data."""
        print("-----------------[Wan2.1 Node] test_encode_decode_random-----------------")
        expected = {
            "GPU": {"latents_mean": 1.0408339500427246, "frames_mean": 0.37298184633255005},
            "NPU": {"latents_mean": 1.0393882989883423, "frames_mean": 0.371329426765441},
        }[CURRENT_PLATFORM]

        seed_g = _make_generator()
        video = torch.randn(IN_DIMS, generator=seed_g, device="cpu", dtype=self.dtype).to(self.device)
        if self.rank == 0:
            print(f"video (random) - {video.shape} {video.min():.4f}~{video.max():.4f}")

        latents, frames = _encode_decode(self.vae, video, self.rank)

        self.assertEqual(list(latents.shape[:2]), [1, LATENT_CHANNELS])
        self.assertEqual(list(frames.shape[:2]), [1, 3])
        if self.rank == 0:
            with self.subTest("latents_mean"):
                self.assertAlmostEqual(
                    latents.float().abs().mean().item(), expected["latents_mean"], places=TEST_EPS_PLACE
                )
            with self.subTest("frames_mean"):
                self.assertAlmostEqual(
                    frames.float().abs().mean().item(), expected["frames_mean"], places=TEST_EPS_PLACE
                )

    SPEED_TOLERANCE_PERCENT = 5

    def test_encode_decode_speed(self):
        ########################################################################
        # P800 各个卡数多卡并行测试耗时参考：640x640x81帧
        # GPU 卡数 :    1卡     2卡      3卡      4卡      6卡      8卡
        # Decoder :   5.43s   3.18s   2.28s    1.98s    1.43s    1.10s
        # Encoder :   3.16s   1.86s   1.35s    1.16s    0.83s    0.65s
        ########################################################################
        # L20 各个卡数多卡并行测试耗时参考：640x640x81帧
        # GPU 卡数 :    1卡     2卡      3卡      4卡
        # Decoder :   9.13s   5.13s    3.53s   2.952s
        # Encoder :   5.30s   2.97s    2.03s   1.697s
        ########################################################################
        """Benchmark VAE encoder and decoder speed."""
        print("-----------------[Wan2.1 Node] test_encode_decode_speed-----------------")
        encode_time_limits = {
            "GPU": {1: 5318, 2: 2980},
            "NPU": {1: 10000, 2: 10000},
        }[CURRENT_PLATFORM]
        decode_time_limits = {
            "GPU": {1: 9130, 2: 5144},
            "NPU": {1: 10000, 2: 10000},
        }[CURRENT_PLATFORM]
        torch.cuda.empty_cache()
        seed_g = _make_generator()
        benchmark_video = torch.randn(IN_DIMS, generator=seed_g, device="cpu", dtype=self.dtype).to(self.device)

        def _encode():
            return self.vae.encode(benchmark_video)

        encoder_avg_ms = _benchmark(_encode, self.device)
        if self.rank == 0:
            print(f"[Encoder] avg cost time: {encoder_avg_ms:.2f} ms (world_size={self.world_size})")

        torch.cuda.empty_cache()

        benchmark_latents = _ensure_tensor(self.vae.encode(benchmark_video))

        def _decode():
            return _ensure_tensor(self.vae.decode(benchmark_latents))

        decoder_avg_ms = _benchmark(_decode, self.device)
        if self.rank == 0:
            print(f"[Decoder] avg cost time: {decoder_avg_ms:.2f} ms (world_size={self.world_size})")
        if self.rank == 0:
            tolerance = 1 + self.SPEED_TOLERANCE_PERCENT / 100
            encoder_limit = encode_time_limits.get(self.world_size, 0) * tolerance
            decoder_limit = decode_time_limits.get(self.world_size, 0) * tolerance
            with self.subTest("encoder_speed"):
                self.assertLess(
                    encoder_avg_ms,
                    encoder_limit,
                    f"Encoder too slow: {encoder_avg_ms:.2f} ms (limit={encoder_limit} ms)",
                )
            with self.subTest("decoder_speed"):
                self.assertLess(
                    decoder_avg_ms,
                    decoder_limit,
                    f"Decoder too slow: {decoder_avg_ms:.2f} ms (limit={decoder_limit} ms)",
                )


if __name__ == "__main__":
    unittest.main()
