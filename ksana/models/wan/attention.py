# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from typing import Any

from ksana.attention import AttentionBackendEnum, get_attn_backend

__all__ = [
    "flash_attention",
    "sage_attention",
    "attention",
]


def _build_backend_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float | None,
    forced_backend: AttentionBackendEnum | None,
) -> Any:
    backend_cls = get_attn_backend(
        head_size=q.size(-1),
        dtype=q.dtype,
        supported_attention_backends=(
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ),
        forced_backend=forced_backend,
    )
    impl_cls = backend_cls.get_impl_cls()
    return impl_cls(
        num_heads=q.size(2),
        head_size=q.size(3),
        causal=causal,
        softmax_scale=softmax_scale if softmax_scale is not None else q.size(3) ** -0.5,
        num_kv_heads=k.size(2),
    )


def _run_backend(
    impl: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: torch.Tensor | None,
    k_lens: torch.Tensor | None,
    dropout_p: float,
    softmax_scale: float | None,
    q_scale: torch.Tensor | None,
    causal: bool,
    window_size: tuple[int, int],
    deterministic: bool,
    version: int | None,
) -> torch.Tensor:
    return impl.forward(
        q,
        k,
        v,
        attn_metadata=None,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        fa_version=version,
    )


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=None,
    version=None,
):
    impl = _build_backend_impl(
        q,
        k,
        causal=causal,
        softmax_scale=softmax_scale,
        forced_backend=AttentionBackendEnum.FLASH_ATTN,
    )
    return _run_backend(
        impl,
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        version=version,
    )


def sage_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=None,
):
    impl = _build_backend_impl(
        q,
        k,
        causal=causal,
        softmax_scale=softmax_scale,
        forced_backend=AttentionBackendEnum.SAGE_ATTN,
    )
    return _run_backend(
        impl,
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        version=None,
    )


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=None,
    fa_version=None,
):
    try:
        impl = _build_backend_impl(
            q,
            k,
            causal=causal,
            softmax_scale=softmax_scale,
            forced_backend=None,
        )
    except RuntimeError:
        impl = _build_backend_impl(
            q,
            k,
            causal=causal,
            softmax_scale=softmax_scale,
            forced_backend=AttentionBackendEnum.TORCH_SDPA,
        )

    return _run_backend(
        impl,
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        version=fa_version,
    )
