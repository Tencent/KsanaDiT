# Copyright 2025 Tencent
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

import sys
import time
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from ksana.accelerator import platform
from ksana.operations.fuse_qkv.fuse_qkv import should_use_qkv_fusion

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ksana.operations.fuse_qkv import QKVProjectionMixin


@unittest.skipIf(not should_use_qkv_fusion(None), "QKV fusion is globally disabled (should_use_qkv_fusion=False)")
class TestQKVProjectionMixinPerformance(unittest.TestCase):
    def _run_mixin_performance_test(
        self,
        scenario_name: str,
        fused_name: str,
        separate_names: tuple,
        bias: bool,
    ):
        dim = 3072
        batch = 1
        device = torch.device("cuda")
        dtype = torch.float16
        num_warmup = 10
        num_runs = 50
        seq_lengths = [1024, 2048, 8192]  # TODO(qiannan): 4096流水线容易被卡，先删掉了

        class FusedMixinModel(nn.Module, QKVProjectionMixin):
            def __init__(self):
                super().__init__()
                self._setup_qkv_projection(
                    dim=dim,
                    operations=type("Operations", (), {"Linear": nn.Linear})(),
                    device=device,
                    dtype=dtype,
                    bias=bias,
                    fused_name=fused_name,
                    separate_names=separate_names,
                )

            def forward(self, x):
                return self.compute_qkv(x)

        class SeparateBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(dim, dim, bias=bias, device=device, dtype=dtype)
                self.k = nn.Linear(dim, dim, bias=bias, device=device, dtype=dtype)
                self.v = nn.Linear(dim, dim, bias=bias, device=device, dtype=dtype)

            def forward(self, x):
                return self.q(x), self.k(x), self.v(x)

        fused_model = FusedMixinModel().eval()
        separate_model = SeparateBaseline().eval()

        fused_layer = getattr(fused_model, fused_name)
        with torch.no_grad():
            fused_layer.weight.copy_(
                torch.cat([separate_model.q.weight, separate_model.k.weight, separate_model.v.weight], dim=0)
            )
            if bias:
                fused_layer.bias.copy_(
                    torch.cat([separate_model.q.bias, separate_model.k.bias, separate_model.v.bias], dim=0)
                )

        print(f"\n=== QKVProjectionMixin Performance Test: {scenario_name} ===")
        for seq_len in seq_lengths:
            x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)

            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = fused_model(x)
                    _ = separate_model(x)
            torch.cuda.synchronize()

            # Benchmark separate
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = separate_model(x)
            torch.cuda.synchronize()
            separate_time = (time.perf_counter() - start) / num_runs * 1000

            # Benchmark fused (via QKVProjectionMixin)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = fused_model(x)
            torch.cuda.synchronize()
            fused_time = (time.perf_counter() - start) / num_runs * 1000

            speedup = separate_time / fused_time
            print(
                f"seq_len={seq_len:5d}: separate={separate_time:.3f}ms, "
                f"mixin_fused={fused_time:.3f}ms, speedup={speedup:.2f}x"
            )
            self.assertLess(
                fused_time,
                separate_time,
                f"QKVProjectionMixin fused should be faster than separate at seq_len={seq_len}",
            )

        x = torch.randn(batch, 512, dim, device=device, dtype=dtype)
        with torch.no_grad():
            q_sep, k_sep, v_sep = separate_model(x)
            q_fused, k_fused, v_fused = fused_model(x)
        torch.testing.assert_close(q_sep, q_fused, rtol=0.01, atol=0.005)
        torch.testing.assert_close(k_sep, k_fused, rtol=0.01, atol=0.005)
        torch.testing.assert_close(v_sep, v_fused, rtol=0.01, atol=0.005)

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available")
    def test_mixin_performance_wan_style(self):
        self._run_mixin_performance_test(
            scenario_name="Wan (single stream, no bias)",
            fused_name="qkv",
            separate_names=("q", "k", "v"),
            bias=False,
        )

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available")
    def test_mixin_performance_qwen_style(self):
        self._run_mixin_performance_test(
            scenario_name="Qwen (image stream, with bias)",
            fused_name="to_qkv",
            separate_names=("to_q", "to_k", "to_v"),
            bias=True,
        )

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available")
    def test_mixin_performance_qwen_dual_stream(self):
        dim = 3072
        batch = 1
        device = torch.device("cuda")
        dtype = torch.float16
        num_warmup = 10
        num_runs = 50
        seq_lengths = [1024, 2048, 4096]

        class QwenDualStreamMixin(nn.Module, QKVProjectionMixin):
            def __init__(self):
                super().__init__()
                # Image stream
                self._setup_qkv_projection(
                    dim=dim,
                    operations=type("Operations", (), {"Linear": nn.Linear})(),
                    device=device,
                    dtype=dtype,
                    bias=True,
                    fused_name="to_qkv",
                    separate_names=("to_q", "to_k", "to_v"),
                    prefix="img_",
                )
                # Text stream
                self._setup_qkv_projection(
                    dim=dim,
                    operations=type("Operations", (), {"Linear": nn.Linear})(),
                    device=device,
                    dtype=dtype,
                    bias=True,
                    fused_name="add_qkv_proj",
                    separate_names=("add_q_proj", "add_k_proj", "add_v_proj"),
                    prefix="txt_",
                )

            def forward(self, img_x, txt_x):
                img_q, img_k, img_v = self.compute_qkv(img_x, prefix="img_")
                txt_q, txt_k, txt_v = self.compute_qkv(txt_x, prefix="txt_")
                return img_q, img_k, img_v, txt_q, txt_k, txt_v

        # Separate baseline for dual stream
        class QwenDualStreamSeparate(nn.Module):
            def __init__(self):
                super().__init__()
                # Image stream
                self.to_q = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
                self.to_k = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
                self.to_v = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
                # Text stream
                self.add_q_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
                self.add_k_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)
                self.add_v_proj = nn.Linear(dim, dim, bias=True, device=device, dtype=dtype)

            def forward(self, img_x, txt_x):
                img_q = self.to_q(img_x)
                img_k = self.to_k(img_x)
                img_v = self.to_v(img_x)
                txt_q = self.add_q_proj(txt_x)
                txt_k = self.add_k_proj(txt_x)
                txt_v = self.add_v_proj(txt_x)
                return img_q, img_k, img_v, txt_q, txt_k, txt_v

        fused_model = QwenDualStreamMixin().eval()
        separate_model = QwenDualStreamSeparate().eval()

        with torch.no_grad():
            fused_model.to_qkv.weight.copy_(
                torch.cat([separate_model.to_q.weight, separate_model.to_k.weight, separate_model.to_v.weight], dim=0)
            )
            fused_model.to_qkv.bias.copy_(
                torch.cat([separate_model.to_q.bias, separate_model.to_k.bias, separate_model.to_v.bias], dim=0)
            )
            fused_model.add_qkv_proj.weight.copy_(
                torch.cat(
                    [
                        separate_model.add_q_proj.weight,
                        separate_model.add_k_proj.weight,
                        separate_model.add_v_proj.weight,
                    ],
                    dim=0,
                )
            )
            fused_model.add_qkv_proj.bias.copy_(
                torch.cat(
                    [separate_model.add_q_proj.bias, separate_model.add_k_proj.bias, separate_model.add_v_proj.bias],
                    dim=0,
                )
            )

        print("\n=== QKVProjectionMixin Performance Test: Qwen Dual Stream ===")
        for seq_len in seq_lengths:
            img_x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
            txt_x = torch.randn(batch, seq_len // 4, dim, device=device, dtype=dtype)  # Text usually shorter

            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = fused_model(img_x, txt_x)
                    _ = separate_model(img_x, txt_x)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = separate_model(img_x, txt_x)
            torch.cuda.synchronize()
            separate_time = (time.perf_counter() - start) / num_runs * 1000

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = fused_model(img_x, txt_x)
            torch.cuda.synchronize()
            fused_time = (time.perf_counter() - start) / num_runs * 1000

            speedup = separate_time / fused_time
            print(
                f"img_seq={seq_len:5d}, txt_seq={seq_len//4:4d}: separate={separate_time:.3f}ms, "
                f"mixin_fused={fused_time:.3f}ms, speedup={speedup:.2f}x"
            )
            self.assertLess(
                fused_time,
                separate_time,
                f"QKVProjectionMixin dual-stream fused should be faster at seq_len={seq_len}",
            )


if __name__ == "__main__":
    unittest.main()
