# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.cuda.nvtx as nvtx

from ksana.operations.attention import AttentionBackendEnum
from ksana.cache import KsanaCache, DBCache
from ksana.utils import time_range, gather_forward, get_rank_id, log


__all__ = ["WanModel"]


def repeat_e(e, x):
    target = x.size(1)
    if e.size(1) == target:
        return e
    repeats = max(1, target // e.size(1))
    expanded = torch.repeat_interleave(e, repeats, dim=1)
    if expanded.size(1) < target:
        expanded = torch.repeat_interleave(e, repeats + 1, dim=1)
    if expanded.size(1) > target:
        expanded = expanded[:, :target]
    return expanded


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast("cuda", enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.q = operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
        self.k = operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
        self.v = operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
        self.o = operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = (
            operation_settings.get("operations").RMSNorm(
                dim, eps=eps, elementwise_affine=True, device=device, dtype=torch.float32
            )
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            operation_settings.get("operations").RMSNorm(
                dim, eps=eps, elementwise_affine=True, device=device, dtype=torch.float32
            )
            if qk_norm
            else nn.Identity()
        )

        self.attention = operation_settings.get("operations").Attn(
            num_heads=num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=(
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.SAGE_ATTN,
                AttentionBackendEnum.TORCH_SDPA,
            ),
        )

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = self.attention(
            rope_apply(q, grid_sizes, freqs),
            rope_apply(k, grid_sizes, freqs),
            v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = self.attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        operation_settings={},
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.norm1 = operation_settings.get("operations").LayerNorm(
            dim, eps, elementwise_affine=False, device=device, dtype=dtype
        )
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings
        )
        self.norm3 = (
            operation_settings.get("operations").LayerNorm(
                dim, eps, elementwise_affine=True, device=device, dtype=torch.float32
            )
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps, operation_settings=operation_settings
        )
        self.norm2 = operation_settings.get("operations").LayerNorm(
            dim, eps, elementwise_affine=False, device=device, dtype=dtype
        )
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=device, dtype=dtype),
        )
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))

    # @nvtx_range
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
            seq_lens=seq_lens, #[7200]
            grid_sizes=grid_sizes, # [[ 2, 45, 80]]
            freqs=self.freqs, #[1024, 64]
            context=context, #[1, 512, 5120]

        Args:
            x(Tensor): Shape [B, L, C]                                                #: [1, 7200, 5120]
            e(Tensor): Shape [B, L1, 6, C]                                            #: [1, 7200, 6, 5120]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W) #: [2, 45, 80]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]                #: [1024, 64]
        """
        # self.modulation : [1, 6, 5120]
        if e.ndim < 4:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=1)
        else:
            e = (self.modulation.unsqueeze(0) + e).unbind(2)
        # [bs, 6, 5120] => [bs, 1, 5120] * 6

        # self-attention
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            seq_lens,
            grid_sizes,
            freqs,
        )
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # cross-attention & ffn function
        # @nvtx_range
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.norm = operation_settings.get("operations").LayerNorm(
            dim, eps, elementwise_affine=False, device=device, dtype=dtype
        )
        self.head = operation_settings.get("operations").Linear(dim, out_dim, device=device, dtype=dtype)
        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=dtype))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        if e.ndim < 3:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(1)).chunk(2, dim=1)
        else:
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).unbind(2)
        x = self.head(torch.addcmul(repeat_e(e[0], x), self.norm(x), 1 + repeat_e(e[1], x)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        operations=None,
        device=None,
        dtype=None,
        sp_size=1,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            sp_size (`int`, *optional*, defaults to 1):
                Sequence parallelism size
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "ti2v", "s2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.sp_size = sp_size

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32
        )
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            operations.Linear(dim, dim, device=device, dtype=dtype),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=device, dtype=dtype))
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    operation_settings=operation_settings,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def set_keep_in_fp32_modules(self):
        self._keep_in_fp32_modules = [
            "patch_embedding",
            "time_embedding",
            "time_projection",
            "head",
            "norm3",
            "norm_q",
            "norm_k",
            "img_emb.proj.0",
            "img_emb.proj.4",
        ]
        self._keep_in_fp32_params = self._find_fp32_params(["modulation"])

    def _find_fp32_params(self, keywords):
        return [name for name, _ in self.named_parameters() if any(keyword in name for keyword in keywords)]

    # @nvtx_range
    def forward(
        self,
        x: torch.Tensor,
        t,
        cache: KsanaCache,
        phase: str,
        context: torch.Tensor,
        seq_len,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                Input video tensor with shape [B, C_in, F, H, W], : [bs, 16, v, h, w]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (Tensor):
                Text embeddings tensor with shape [B, L, C] : [bs, 73, 4096]
            seq_len (`int`): 7200
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, F, H / 8, W / 8]
        """
        # x [bs, 16, f, h, w], y [bs, 20, f, h, w]
        if self.model_type == "i2v" and y is None:
            raise ValueError("y must be provided for i2v model")
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            # x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = torch.cat([x, y.to(x.dtype)], dim=1)
        # embeddings
        # [bs, 16, fi, hi, wi] => [bs, 5120, f, h, w]
        x = self.patch_embedding(x.float()).to(x.dtype)
        bs = x.shape[0]
        # grid_sizes: 支持 batch - 所有样本共享相同的 grid (假设 batch 内尺寸相同)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
        # [bs, 5120, f*h*w] => [bs, f*h*w, 5120]
        x = x.flatten(2).transpose(1, 2)
        # seq_lens: 支持 batch
        seq_lens = torch.full((bs,), x.shape[1], dtype=torch.int32, device=x.device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [x, x.new_zeros(x.size(0), seq_len - x.size(1), x.size(2))], dim=1
        )  # pad f*h*w to seqlen, => [bs, seqlen, 5120]

        # time embeddings
        nvtx.range_push("time_embedding")
        # 取第一个 timestep，因为一个batch里的timestamp都是一样的
        timestep = t[0].item() if t.numel() > 1 else t.item()

        bs = t.size(0)  # batch size
        if t.dim() == 1:
            t = t.unsqueeze(1)
        one = t.size(1)  # = 1
        # t: [bs]
        t = t.flatten()
        # freq_dim : 256
        # t: [bs] => [bs, freq_dim:256]
        e = sinusoidal_embedding_1d(self.freq_dim, t)
        # [bs, freq_dim] => [bs, one, freq_dim]
        e = e.unflatten(0, (bs, one))
        # [bs, one, freq_dim=>self.dim:5120]
        e = self.time_embedding(e.to(x.dtype))
        # [bs, one, 5120] => [bs, one, 6*5120]
        e0 = self.time_projection(e)
        # [bs, one, 6*5120] => [bs, one, 6, 5120]
        e0 = e0.unflatten(2, (6, self.dim))
        nvtx.range_pop()

        # context
        context_lens = None
        nvtx.range_push("text_embedding")
        # [bs, 512, 4096] pad to => [bs, text_len:512, 4096]
        padded_context = torch.cat(
            [context, context.new_zeros(bs, self.text_len - context.size(1), context.size(2))], dim=1
        )
        # [bs, text_len, 4096] => [bs, text_len, 5120]
        context = self.text_embedding(padded_context)
        nvtx.range_pop()

        if self.sp_size > 1:
            x = torch.chunk(x, self.sp_size, dim=1)[get_rank_id()]

        # arguments
        kwargs = dict(
            e=e0,  # [bs, seqlen, 6, 5120]
            seq_lens=seq_lens,  # [bs]
            grid_sizes=grid_sizes,  # [bs, 3]
            freqs=self.freqs,  # [1024, 64]
            context=context,  # [bs, text_len:512, 5120]
            context_lens=context_lens,
        )

        if cache is None:
            for block in self.blocks:
                # x: [bs, seqlen, 5120]
                x = block(x, **kwargs)
        else:
            # if cache.need_compile_cache:
            #     x_ori = x.clone()
            #     nvtx.range_push("blocks")
            #     for block in self.blocks:
            #         x = block(x, **kwargs)
            #     nvtx.range_pop()
            #     x_diff = x - x_ori
            #     cache.compile_config_add(timestep, x_diff)
            # TODO(jasonbsun): add unified cache interface
            if isinstance(cache, DBCache):
                # DBCache: FnBn block-wise caching logic
                x = self._forward_with_dbcache(x, cache, phase, timestep, **kwargs)
            else:
                # DCache: full-block caching logic
                x = self._forward_with_dcache(x, cache, phase, timestep, **kwargs)

        # torch.save(x.cpu().abs().mean(dim = len(x.shape) - 1), f"{save_prefix}_xo_mean.pt")

        # head
        # [bs, seqlen, 5120] => [bs, seqlen, 64], e:[bs, seqlen, freq_dim=>self.dim:5120]
        x = self.head(x, e)
        if self.sp_size > 1:
            x = gather_forward(x, dim=1)
        # unpatchify
        # [bs, seqlen, 64] => [bs, 16, fi, hi, wi]
        x = self.unpatchify(x, grid_sizes)
        return x

    def _forward_with_dcache(self, x, cache, phase, timestep, **kwargs):
        use_cache = False
        x_diff = None
        if cache.can_use_cache(phase, x, timestep):
            x_diff = cache.try_get_prev_cache(phase, x, timestep)
            use_cache = x_diff is not None

        if use_cache:
            x = x + x_diff.to(x.device)
        else:
            x_ori = cache.clone_input_x(timestep, x)
            for block in self.blocks:
                x = block(x, **kwargs)
            cache.update_states(phase, timestep, x_ori, x)
        return x

    def _forward_with_dbcache(self, x, cache, phase, timestep, **kwargs):
        use_cache = False
        step = cache.context.current_step
        Fn_start, Fn_end = cache.context.Fn_blocks_range
        Mn_start, Mn_end = cache.context.Mn_blocks_range
        Bn_start, Bn_end = cache.context.Bn_blocks_range

        if cache.can_use_cache(phase, x, timestep):
            # Step 1: Always compute Fn blocks first
            x_ori = x.clone()
            for i in range(Fn_start, min(Fn_end, len(self.blocks))):
                x = self.blocks[i](x, **kwargs)

            # Calculate Fn residual for diff comparison
            Fn_residual = x - x_ori

            # Step 2: Check L1 diff to decide whether to use cache
            if cache.compute_diff_and_decide(phase, Fn_residual):
                # Cache HIT: Use cached Mn+Bn residual
                Bn_residual, _ = cache.try_get_prev_cache(phase, x, timestep)
                if Bn_residual is not None:
                    x = x + Bn_residual.to(x.device)
                    use_cache = True
                    log.info(
                        f"[Wan][DBCache] step={step} phase={phase} cache=HIT "
                        f"skip Mn[{Mn_start},{Mn_end}) Bn[{Bn_start},{Bn_end})"
                    )
                else:
                    log.info(f"[Wan][DBCache] step={step} phase={phase} cache=MISS " f"reason=no_cached_residual")

            if not use_cache:
                # Cache MISS: Compute remaining Mn and Bn blocks
                x_before_Mn = x.clone()

                # Compute Mn blocks
                for i in range(Mn_start, min(Mn_end, len(self.blocks))):
                    x = self.blocks[i](x, **kwargs)

                # Compute Bn blocks
                for i in range(Bn_start, min(Bn_end, len(self.blocks))):
                    x = self.blocks[i](x, **kwargs)

                # Update cache with new residuals
                Bn_residual = x - x_before_Mn
                cache.update_states(phase, timestep, Fn_residual, Bn_residual)
                log.info(
                    f"[Wan][DBCache] step={step} phase={phase} cache=MISS "
                    f"computed Mn[{Mn_start},{Mn_end}) Bn[{Bn_start},{Bn_end})"
                )
        else:
            # Warmup or forced compute: run all blocks
            x_ori = x.clone()
            for block in self.blocks:
                x = block(x, **kwargs)
            # Still update cache for subsequent steps
            # Use a simplified approach: treat entire residual as both Fn and Bn residual
            full_residual = x - x_ori
            cache.update_states(phase, timestep, full_residual, full_residual)
            log.info(
                f"[Wan][DBCache] step={step} phase={phase} cache=MISS "
                f"warmup compute all blocks ({len(self.blocks)})"
            )
        # Advance cache step once per diffusion timestep (guarded inside DBCache)
        if hasattr(cache, "advance_step_once"):
            cache.advance_step_once(timestep)
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        b = x.shape[0]  # batch size
        # grid_sizes: [bs, 3] => [f, h, w]
        grid_sizes = grid_sizes[0].tolist() if grid_sizes.dim() > 1 else grid_sizes.tolist()
        # 1. 取前prod(grid_sizes)个patch，reshape为[B, F_patches, H_patches, W_patches, p_f, p_h, p_w, C_out]
        u = x[:, : math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        # 2. 维度重排：从[B, F_patches, H_patches, W_patches, p_f, p_h, p_w, C_out]
        # 转为[B, C_out, F_patches, p_f, H_patches, p_h, W_patches, p_w]
        u = torch.einsum("bfhwpqrc->bcfphqwr", u)
        # 3. 重塑为[B, C_out, F_patches * p_f, H_patches * p_h, W_patches * p_w]
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u

    @time_range
    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
