from __future__ import annotations

from typing import Any, Optional

import torch

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_2_func

    _FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_2_func = None  # type: ignore[assignment]
    _FLASH_ATTN_2_AVAILABLE = False

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_3_func

    _FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_3_func = None  # type: ignore[assignment]
    _FLASH_ATTN_3_AVAILABLE = False

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def is_available() -> bool:
        return _FLASH_ATTN_2_AVAILABLE or _FLASH_ATTN_3_AVAILABLE

    @staticmethod
    def supports(head_size: int, dtype: torch.dtype) -> bool:
        return FlashAttentionBackend.is_available() and torch.cuda.is_available()


class FlashAttentionImpl(AttentionImpl[AttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        **extra_impl_args: Any,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.prefix = prefix

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata],
        *,
        q_lens: Optional[torch.Tensor] = None,
        k_lens: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        q_scale: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        window_size: tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        fa_version: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        if not FlashAttentionBackend.is_available():
            raise RuntimeError("FlashAttention backend requested but not installed.")

        device = query.device
        if device.type != "cuda":
            raise AssertionError("FlashAttention requires CUDA tensors.")

        softmax_scale = softmax_scale if softmax_scale is not None else self.softmax_scale
        causal = self.causal if causal is None else causal

        half_dtypes = (torch.float16, torch.bfloat16)
        original_dtype = query.dtype
        if original_dtype in half_dtypes:
            target_dtype = original_dtype
        else:
            target_dtype = (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            )

        batch, seqlen_q, num_heads, head_dim = query.shape
        seqlen_k = key.shape[1]
        out_dtype = original_dtype

        def _to_target_dtype(t: torch.Tensor) -> torch.Tensor:
            return t if t.dtype == target_dtype else t.to(target_dtype)

        if q_scale is not None:
            scaled_q = q_scale.to(query.dtype) if isinstance(q_scale, torch.Tensor) else q_scale
            query = query * scaled_q

        query = _to_target_dtype(query)
        key = _to_target_dtype(key)
        value = _to_target_dtype(value)

        if q_lens is None:
            q_lens = torch.full((batch,), seqlen_q, dtype=torch.int32, device=device, requires_grad=False)
            flat_q = query.flatten(0, 1)
        else:
            q_lens = q_lens.to(device=device, dtype=torch.int32)
            flat_q = torch.cat([query[i, : int(q_lens[i])].contiguous() for i in range(batch)], dim=0)

        if k_lens is None:
            k_lens = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device, requires_grad=False)
            flat_k = key.flatten(0, 1)
            flat_v = value.flatten(0, 1)
        else:
            k_lens = k_lens.to(device=device, dtype=torch.int32)
            flat_k = torch.cat([key[i, : int(k_lens[i])].contiguous() for i in range(batch)], dim=0)
            flat_v = torch.cat([value[i, : int(k_lens[i])].contiguous() for i in range(batch)], dim=0)

        cumulative_q = torch.cat([q_lens.new_zeros(1), q_lens]).cumsum(0, dtype=torch.int32)
        cumulative_k = torch.cat([k_lens.new_zeros(1), k_lens]).cumsum(0, dtype=torch.int32)
        cumulative_q = cumulative_q.to(device=device)
        cumulative_k = cumulative_k.to(device=device)

        if (fa_version is None or fa_version == 3) and _FLASH_ATTN_3_AVAILABLE:
            attn_out = flash_attn_3_func(  # type: ignore[misc]
                q=flat_q,
                k=flat_k,
                v=flat_v,
                cu_seqlens_q=cumulative_q,
                cu_seqlens_k=cumulative_k,
                max_seqlen_q=int(seqlen_q),
                max_seqlen_k=int(seqlen_k),
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0]
        else:
            if not _FLASH_ATTN_2_AVAILABLE:
                raise RuntimeError("flash_attn module not found. Install FlashAttention or disable this backend.")
            attn_out = flash_attn_2_func(  # type: ignore[misc]
                q=flat_q,
                k=flat_k,
                v=flat_v,
                cu_seqlens_q=cumulative_q,
                cu_seqlens_k=cumulative_k,
                max_seqlen_q=int(seqlen_q),
                max_seqlen_k=int(seqlen_k),
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )

        attn_out = attn_out.unflatten(0, (batch, seqlen_q))
        return attn_out.to(out_dtype)
