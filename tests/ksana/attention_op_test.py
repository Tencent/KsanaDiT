import unittest

import torch
import torch.nn.functional as F

from ksana.accelerator import platform
from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig
from ksana.operations.attention import pick_attn_op

B, L, H, D = 2, 32, 64, 128  # batch, seq_len, num_heads, head_dim


def _make_inputs(device: torch.device | str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(B, L, H, D, device=device, dtype=dtype)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)
    return q, k, v


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Baseline attention computed via PyTorch scaled_dot_product_attention."""
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


class TestLocalAttentionOpBackends(unittest.TestCase):
    @unittest.skipIf(not (platform.is_gpu() or platform.is_npu()), "CUDA/NPU not available.")
    def test_sdpa_accelerator(self) -> None:
        """SDPA backend should run on CUDA or NPU with basic correctness checks."""
        device_type = "cuda" if platform.is_gpu() else "npu"
        device = torch.device(device_type)
        q, k, v = _make_inputs(device=device, dtype=torch.float16)

        # Set deterministic seed for reproducible logs
        torch.manual_seed(0)
        attn = pick_attn_op(KsanaAttentionConfig(backend=KsanaAttentionBackend.TORCH_SDPA))
        attn = attn(
            num_heads=H,
            head_size=D,
            causal=False,
        )

        out = attn(q, k, v)

        print(f"[SDPA] backend={attn.backend_type}, device={q.device}, dtype={q.dtype}, " f"shape={tuple(q.shape)}")

        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available.")
    def test_flash_attn_cuda(self) -> None:
        """FlashAttention backend should initialize and run on CUDA."""
        device = torch.device("cuda")
        q, k, v = _make_inputs(device=device, dtype=torch.float16)

        torch.manual_seed(0)
        attn = pick_attn_op(KsanaAttentionConfig(backend=KsanaAttentionBackend.FLASH_ATTN))
        attn = attn(
            num_heads=H,
            head_size=D,
            causal=False,
        )

        out = attn(q, k, v)

        print(
            f"[FLASH_ATTN] backend={attn.backend_type}, device={q.device}, dtype={q.dtype}, " f"shape={tuple(q.shape)}"
        )

        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available.")
    def test_sage_attn_cuda(self) -> None:
        """SageAttention backend should initialize and run on CUDA."""
        device = torch.device("cuda")
        q, k, v = _make_inputs(device=device, dtype=torch.float16)

        torch.manual_seed(0)
        attn = pick_attn_op(KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN))
        attn = attn(
            num_heads=H,
            head_size=D,
            causal=False,
        )

        out = attn(q, k, v)

        print(
            f"[SAGE_ATTN] backend={attn.backend_type}, device={q.device}, dtype={q.dtype}, " f"shape={tuple(q.shape)}"
        )

        self.assertEqual(out.shape, q.shape)
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available.")
    def test_flash_attn_accuracy_vs_sdpa(self) -> None:
        """FlashAttention output should be numerically close to SDPA baseline."""
        device = torch.device("cuda")
        q, k, v = _make_inputs(device=device, dtype=torch.float16)

        torch.manual_seed(0)

        # Baseline with PyTorch SDPA on the same device / dtype
        ref = _sdpa_reference(q, k, v, causal=False)

        attn = pick_attn_op(KsanaAttentionConfig(backend=KsanaAttentionBackend.FLASH_ATTN))
        attn = attn(
            num_heads=H,
            head_size=D,
            causal=False,
        )
        out = attn(q, k, v)

        diff = (out - ref).abs()
        print(
            "[FLASH_ATTN][accuracy] "
            f"backend={attn.backend_type}, max_diff={diff.max().item():.4e}, "
            f"mean_diff={diff.mean().item():.4e}"
        )

        self.assertEqual(out.shape, ref.shape)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)

    @unittest.skipIf(not platform.is_gpu(), "CUDA not available.")
    def test_sage_attn_accuracy_vs_sdpa(self) -> None:
        """SageAttention output should be numerically close to SDPA baseline."""
        device = torch.device("cuda")
        q, k, v = _make_inputs(device=device, dtype=torch.float16)

        torch.manual_seed(0)

        # Baseline with PyTorch SDPA on the same device / dtype
        ref = _sdpa_reference(q, k, v, causal=False)

        attn = pick_attn_op(KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN))
        attn = attn(
            num_heads=H,
            head_size=D,
            causal=False,
        )
        out = attn(q, k, v)

        diff = (out - ref).abs()
        print(
            "[SAGE_ATTN][accuracy] "
            f"backend={attn.backend_type}, max_diff={diff.max().item():.4e}, "
            f"mean_diff={diff.mean().item():.4e}"
        )

        self.assertEqual(out.shape, ref.shape)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        self.assertLess(max_diff, 1e-1)
        self.assertLess(mean_diff, 1e-2)


if __name__ == "__main__":
    unittest.main()
