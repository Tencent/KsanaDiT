# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

from ksana.utils.rope import apply_comfyui_rope, apply_default_rope
from ..utils import get_rank_id, get_world_size
from .ulysses import distributed_attention


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, rope_func="default"):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    run_dtype = x.dtype

    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)

    sp_rank = get_rank_id()
    sp_size = get_world_size()
    if rope_func == "comfy":
        q = apply_comfyui_rope(q, freqs, sp_rank=sp_rank, sp_size=sp_size)
        k = apply_comfyui_rope(k, freqs, sp_rank=sp_rank, sp_size=sp_size)
    else:
        q = apply_default_rope(q, grid_sizes, freqs, sp_rank=sp_rank, sp_size=sp_size)
        k = apply_default_rope(k, grid_sizes, freqs, sp_rank=sp_rank, sp_size=sp_size)

    x = distributed_attention(
        q.to(run_dtype),
        k.to(run_dtype),
        v.to(run_dtype),
        seq_lens,
        window_size=self.window_size,
        attn_func=self.attention,
    )

    x = x.flatten(2)
    x = self.o(x)
    return x
