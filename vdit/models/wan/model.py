# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention
import torch.cuda.nvtx as nvtx

from vdit.cache import vDitCache
from vdit.utils import time_range

# import ipdb

__all__ = ["WanModel"]


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
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, comfy_operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        if comfy_operation_settings:
            device = comfy_operation_settings.get("device")
            dtype = comfy_operation_settings.get("dtype")
            self.q = comfy_operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
            self.k = comfy_operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
            self.v = comfy_operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
            self.o = comfy_operation_settings.get("operations").Linear(dim, dim, device=device, dtype=dtype)
            self.norm_q = (
                comfy_operation_settings.get("operations").RMSNorm(
                    dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype
                )
                if qk_norm
                else nn.Identity()
            )
            self.norm_k = (
                comfy_operation_settings.get("operations").RMSNorm(
                    dim, eps=eps, elementwise_affine=True, device=device, dtype=dtype
                )
                if qk_norm
                else nn.Identity()
            )
        else:
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.o = nn.Linear(dim, dim)
            self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
            self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

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

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
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
        x = flash_attention(q, k, v, k_lens=context_lens)

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
        comfy_operation_settings={},
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
        if comfy_operation_settings:
            device = comfy_operation_settings.get("device")
            dtype = comfy_operation_settings.get("dtype")
            self.norm1 = comfy_operation_settings.get("operations").LayerNorm(
                dim, eps, elementwise_affine=False, device=device, dtype=dtype
            )
            self.self_attn = WanSelfAttention(
                dim, num_heads, window_size, qk_norm, eps, comfy_operation_settings=comfy_operation_settings
            )
            self.norm3 = (
                comfy_operation_settings.get("operations").LayerNorm(
                    dim, eps, elementwise_affine=True, device=device, dtype=dtype
                )
                if cross_attn_norm
                else nn.Identity()
            )
            self.cross_attn = WanCrossAttention(
                dim, num_heads, (-1, -1), qk_norm, eps, comfy_operation_settings=comfy_operation_settings
            )
            self.norm2 = comfy_operation_settings.get("operations").LayerNorm(
                dim, eps, elementwise_affine=False, device=device, dtype=dtype
            )
            self.ffn = nn.Sequential(
                comfy_operation_settings.get("operations").Linear(dim, ffn_dim, device=device, dtype=dtype),
                nn.GELU(approximate="tanh"),
                comfy_operation_settings.get("operations").Linear(ffn_dim, dim, device=device, dtype=dtype),
            )
            self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=device, dtype=dtype))
        else:
            self.norm1 = WanLayerNorm(dim, eps)
            self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
            self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
            self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
            self.norm2 = WanLayerNorm(dim, eps)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(ffn_dim, dim),
            )
            # modulation
            self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

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
        # print(f"-----------x:{x.cpu().abs().mean().item()}")
        # TODO: remove fp32 assert
        # assert e.dtype == torch.float32
        # ipdb.set_trace()
        # with torch.amp.autocast("cuda", dtype=torch.float32):
        # self.modulation : [1, 6, 5120]
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # [bs, seqlen, 6, 5120] => [bs, seqlen, 1, 5120] * 6
        # assert e[0].dtype == torch.float32

        # ipdb.set_trace()
        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens,
            grid_sizes,
            freqs,
        )
        # with torch.amp.autocast("cuda", dtype=torch.float32):
        x = x + y * e[2].squeeze(2)
        del y

        # cross-attention & ffn function
        # @nvtx_range
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            # with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, comfy_operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        if comfy_operation_settings:
            device = comfy_operation_settings.get("device")
            dtype = comfy_operation_settings.get("dtype")
            self.norm = comfy_operation_settings.get("operations").LayerNorm(
                dim, eps, elementwise_affine=False, device=device, dtype=dtype
            )
            self.head = comfy_operation_settings.get("operations").Linear(dim, out_dim, device=device, dtype=dtype)
            # modulation
            self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=device, dtype=dtype))
            print(f"head-----------dtype:{dtype}")
        else:
            self.norm = WanLayerNorm(dim, eps)
            self.head = nn.Linear(dim, out_dim)
            # modulation
            self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        # use comfy.model_management.cast_to
        # with torch.amp.autocast("cuda", dtype=torch.float32):
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        print(f"-----------x.dtype{x.dtype}, headoutput:{x.cpu().abs().mean().item()}")
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
        disable_weight_init_operations=None,
        device=None,
        dtype=None,
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

        # embeddings
        if disable_weight_init_operations:
            self.patch_embedding = disable_weight_init_operations.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=torch.float32
            )
            self.text_embedding = nn.Sequential(
                disable_weight_init_operations.Linear(text_dim, dim, device=device, dtype=dtype),
                nn.GELU(approximate="tanh"),
                disable_weight_init_operations.Linear(dim, dim, device=device, dtype=dtype),
            )
            self.time_embedding = nn.Sequential(
                disable_weight_init_operations.Linear(freq_dim, dim, device=device, dtype=dtype),
                nn.SiLU(),
                disable_weight_init_operations.Linear(dim, dim, device=device, dtype=dtype),
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), disable_weight_init_operations.Linear(dim, dim * 6, device=device, dtype=dtype)
            )
        else:
            self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
            )

            self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        comfy_operation_settings = {"operations": disable_weight_init_operations, "device": device, "dtype": dtype}

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
                    comfy_operation_settings=comfy_operation_settings,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps, comfy_operation_settings=comfy_operation_settings)

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

        # initialize weights
        if disable_weight_init_operations is None:
            self.init_weights()

    # @nvtx_range
    def forward(
        self,
        x: torch.Tensor,
        t,
        cache: vDitCache,
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
        # ipdb.set_trace()

        if self.model_type == "i2v":
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            # x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = torch.cat([x, y], dim=0)
        # embeddings
        # [bs, 16, fi, hi, wi] => [bs, 5120, f, h, w]
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = torch.stack([torch.tensor(x.shape[2:], dtype=torch.long)])  # => [f, h, w]
        # [bs, 5120, f*h*w] => [bs, f*h*w, 5120]
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([x.shape[1]], dtype=torch.long)  # seqlen
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [x, x.new_zeros(x.size(0), seq_len - x.size(1), x.size(2))], dim=1
        )  # pad f*h*w to seqlen, => [bs, seqlen, 5120]

        # time embeddings
        nvtx.range_push("time_embedding")
        timestep = t.item()  # TODO: support bs > 1
        # TODO: e support do not expand
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)  # => [bs, seqlen]

        bs = t.size(0)  # 1
        # TODO: e do not flatten and support bs > 1
        t = t.flatten()  # [7200]
        # freq_dim : 256
        # t: [seqlen] => [seqlen, freq_dim:256]
        e = sinusoidal_embedding_1d(self.freq_dim, t)
        # [seqlen, freq_dim] => [bs, seqlen, freq_dim]
        e = e.unflatten(0, (bs, seq_len))
        # [bs, seqlen, freq_dim=>self.dim:5120]
        e = self.time_embedding(e.to(x.dtype))
        # [bs, seqlen, 5120] => [bs, seqlen, 6*5120]
        e0 = self.time_projection(e)
        # [bs, seqlen, 6*5120] => [bs, seqlen, 6, 5120]
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

        # arguments
        kwargs = dict(
            e=e0,  # [bs, seqlen, 6, 5120]
            seq_lens=seq_lens,  # [seqlen]
            grid_sizes=grid_sizes,  # [[f, h, w]]
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
            # else:
            use_cache = False
            if cache.can_use_cache(phase, x, timestep):
                x_diff = cache.try_get_prev_cache(phase, x, timestep)
                use_cache = x_diff is not None
            if use_cache:
                x = x + x_diff
                # cache.post_cacheprocess(phase, timestep, x_diff)
            else:
                x_ori = x.clone()
                # nvtx.range_push("blocks")
                for block in self.blocks:
                    # x: [bs, seqlen, 5120]
                    x = block(x, **kwargs)
                # nvtx.range_pop()
                cache.update_states(phase, timestep, x_ori, x)

        # torch.save(x.cpu().abs().mean(dim = len(x.shape) - 1), f"{save_prefix}_xo_mean.pt")

        # head
        # [bs, seqlen, 5120] => [bs, seqlen, 64], e:[bs, seqlen, freq_dim=>self.dim:5120]
        x = self.head(x, e)

        # unpatchify
        # TODO: support bs > 1
        # [1, seqlen, 64] => [16, fi, hi, wi]
        x = self.unpatchify(x, grid_sizes)
        x = x[0].unsqueeze(0)

        # => [bs, 16, fi, hi, wi]
        print(f"--------shape:{x.shape}---lastout:{x.cpu().abs().mean().item()}")
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
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

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
