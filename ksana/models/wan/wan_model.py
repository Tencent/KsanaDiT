# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from ksana.accelerator import platform
from ksana.cache import KsanaHybridCache
from ksana.config import KsanaFETAConfig, KsanaSLGConfig
from ksana.operations.fuse_qkv import QKVProjectionMixin
from ksana.utils import all_to_all, gather_forward, get_rank_id, get_world_size, time_range
from ksana.utils.experimental_sampling import compute_feta_scores
from ksana.utils.rope import EmbedND, apply_comfyui_rope, apply_default_rope

if platform.is_npu():
    import torch_npu  # pylint: disable=unused-import # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # pylint: disable=unused-import # noqa: F401


__all__ = ["WanModel", "VaceWanModel"]


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


class WanBaseAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, operation_settings=None):
        super().__init__()
        if operation_settings is None:
            operation_settings = {}
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        ops = operation_settings.get("operations")
        rms_dtype = operation_settings.get("rms_dtype")

        self.o = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.norm_q = (
            ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=torch.float32, rms_dtype=rms_dtype)
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            ops.RMSNorm(dim, eps=eps, elementwise_affine=True, device=device, dtype=torch.float32, rms_dtype=rms_dtype)
            if qk_norm
            else nn.Identity()
        )
        self.attention = ops.Attn(num_heads=num_heads, head_size=self.head_dim, causal=False)


class WanSelfAttention(WanBaseAttention, QKVProjectionMixin):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=None,
        qk_norm=True,
        eps=1e-6,
        block_id=-1,
        operation_settings=None,
        enable_sla=False,
    ):
        if window_size is None:
            window_size = (-1, -1)
        if operation_settings is None:
            operation_settings = {}

        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings)
        self.block_id = block_id

        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        ops = operation_settings.get("operations")

        self._setup_qkv_projection(
            dim=dim,
            operations=ops,
            device=device,
            dtype=dtype,
            bias=True,
            fused_name="qkv",
            separate_names=("q", "k", "v"),
        )

        if enable_sla:
            self.proj_l = ops.Linear(self.head_dim, self.head_dim, device=device, dtype=dtype)

        self.sp_rank = get_rank_id()
        self.sp_size = get_world_size()

    def _gather_qkv(self, q, k, v):
        if self.sp_size == 1:
            return q, k, v
        # all to all
        q = all_to_all(q, scatter_dim=2, gather_dim=1)
        k = all_to_all(k, scatter_dim=2, gather_dim=1)
        v = all_to_all(v, scatter_dim=2, gather_dim=1)
        return q, k, v

    def _gather_attn_output(self, x):
        if self.sp_size == 1:
            return x
        return all_to_all(x, scatter_dim=1, gather_dim=2)

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        step_iter,
        latent_shape,
        rope_func="default",
        feta_enabled: bool = False,
        feta_weight: float = 2.0,
        num_frames: int = 1,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q, k, v = self.compute_qkv(x)
        q = self.norm_q(q).view(b, s, n, d)
        k = self.norm_k(k).view(b, s, n, d)
        v = v.view(b, s, n, d)

        # Apply RoPE
        if rope_func == "comfy":
            q = apply_comfyui_rope(q, freqs, sp_rank=self.sp_rank, sp_size=self.sp_size)
            k = apply_comfyui_rope(k, freqs, sp_rank=self.sp_rank, sp_size=self.sp_size)
        else:
            q = apply_default_rope(q, grid_sizes, freqs, sp_rank=self.sp_rank, sp_size=self.sp_size)
            k = apply_default_rope(k, grid_sizes, freqs, sp_rank=self.sp_rank, sp_size=self.sp_size)

        feta_scores = None
        if feta_enabled and num_frames > 1:
            feta_scores = compute_feta_scores(q, k, num_frames, feta_weight)

        q, k, v = self._gather_qkv(q, k, v)

        kwargs = {
            "k_lens": seq_lens,
            "window_size": self.window_size,
            "latent_shape": list(latent_shape),
            "step_iter": step_iter,
            "block_id": self.block_id,
            "proj_l": self.proj_l if hasattr(self, "proj_l") else None,
        }
        x = self.attention(q, k, v, **kwargs)

        x = self._gather_attn_output(x)

        if feta_scores is not None:
            x = x * feta_scores

        x = x.flatten(2)
        if hasattr(self.o, "weight"):
            w_dtype = self.o.weight.dtype
            if w_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if x.dtype != torch.float16:
                    x = x.to(torch.float16)
            elif x.dtype != w_dtype:
                x = x.to(w_dtype)
        x = self.o(x)
        return x


class WanCrossAttention(WanBaseAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        operation_settings=None,
        enable_sla=False,  # unused, for interface consistency
    ):
        if operation_settings is None:
            operation_settings = {}

        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings)

        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        ops = operation_settings.get("operations")

        self.q = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.k = ops.Linear(dim, dim, device=device, dtype=dtype)
        self.v = ops.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = self.attention(q, k, v, k_lens=context_lens, dense_only=True)

        x = x.flatten(2)
        if hasattr(self.o, "weight"):
            w_dtype = self.o.weight.dtype
            if w_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                if x.dtype != torch.float16:
                    x = x.to(torch.float16)
            elif x.dtype != w_dtype:
                x = x.to(w_dtype)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        block_id,
        window_size=None,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        operation_settings=None,
        is_turbo_diffusion_wan_model=False,
    ):
        if window_size is None:
            window_size = (-1, -1)
        if operation_settings is None:
            operation_settings = {}
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.block_id = block_id

        # layers
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.norm1 = operation_settings.get("operations").LayerNorm(
            dim, eps, elementwise_affine=False, device=device, dtype=dtype
        )
        self.self_attn = WanSelfAttention(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            block_id=block_id,
            operation_settings=operation_settings,
            enable_sla=is_turbo_diffusion_wan_model,
        )
        self.norm3 = (
            operation_settings.get("operations").LayerNorm(
                dim, eps, elementwise_affine=True, device=device, dtype=torch.float32
            )
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
            operation_settings=operation_settings,
            enable_sla=False,
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
        step_iter=None,
        latent_shape=None,
        rope_func="default",
        feta_enabled: bool = False,
        feta_weight: float = 2.0,
        num_frames: int = 1,
    ):
        r"""
            seq_lens=seq_lens, # [7200]
            grid_sizes=grid_sizes, # [[ 2, 45, 80]]
            freqs=self.freqs, # [1024, 64]
            context=context, # [1, 512, 5120]

        Args:
            x(Tensor): Shape [B, L, C] - Input hidden states
            e(Tensor): Shape [B, L1, 6, C] - Time embedding modulation
            seq_lens(Tensor): Shape [B] - Length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3] - (F, H, W) grid dimensions
            freqs(Tensor): RoPE frequencies, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L2, C] - Text embeddings for cross-attention
            context_lens(Tensor): Shape [B] - Length of context sequences
            step_iter(int): Current step iteration
            latent_shape: Shape of the latent tensor
            rope_func(str): RoPE function to use
            feta_enabled(bool): Whether to apply FETA enhancement in self-attention
            feta_weight(float): FETA enhancement weight (higher = stronger smoothing)
            num_frames(int): Number of video frames for FETA computation

        Returns:
            Tensor: Output hidden states with shape [B, L, C]
        """
        # self.modulation : [1, 6, 5120]
        if e.ndim < 4:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=1)
        else:
            e = (self.modulation.unsqueeze(0) + e).unbind(2)
        # [bs, 6, 5120] => [bs, 1, 5120] * 6

        # self-attention with optional FETA enhancement
        y = self.self_attn(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            seq_lens,
            grid_sizes,
            freqs,
            step_iter=step_iter,
            rope_func=rope_func,
            latent_shape=latent_shape,
            feta_enabled=feta_enabled,
            feta_weight=feta_weight,
            num_frames=num_frames,
        )
        x = torch.addcmul(x, y, repeat_e(e[2], x))
        del y

        # cross-attention & ffn function
        # @nvtx_range
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm2(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class VaceWanAttentionBlock(WanAttentionBlock):
    """
    VACE-specific attention block for processing vace_context.
    Has before_proj at block_id=0 and after_proj for all blocks.
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        block_id=0,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        operation_settings=None,
    ):
        if operation_settings is None:
            operation_settings = {}
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            block_id,
            window_size,
            qk_norm,
            cross_attn_norm,
            eps,
            operation_settings=operation_settings,
        )

        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        operations = operation_settings.get("operations")

        if block_id == 0:
            self.before_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.after_proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, c, x=None, **kwargs):
        """
        VACE block forward.

        Args:
            c: VACE context tensor [B, L, C]
            x: Main latent tensor (only used at block_id=0)
            **kwargs: Other arguments passed to parent forward (includes FETA params)

        Returns:
            c_skip: Skip connection tensor to be added to main branch
            c: Updated context tensor for next VACE block
        """
        if self.block_id == 0 and x is not None:
            c = self.before_proj(c) + x

        # Call parent forward (WanAttentionBlock)
        c = super().forward(c, **kwargs)

        # Generate skip connection
        c_skip = self.after_proj(c)
        return c_skip, c


class BaseWanVaceAttentionBlock(WanAttentionBlock):
    """
    Main model attention block with VACE hints injection support.

    Inherits FETA support from WanAttentionBlock.
    SLG (Skip Layer Guidance) is handled at the model level by checking block_id.
    """

    def forward(self, x, vace_hints=None, vace_context_scale=1.0, **kwargs):
        """
        Forward pass with VACE hints injection.

        Args:
            x: Input tensor [B, L, C]
            vace_hints: List of VACE hint tensors from VACE blocks
            vace_context_scale: Scale factor for VACE hints injection
            **kwargs: Other arguments including FETA params passed to parent

        Returns:
            Tensor: Output with VACE hints injected
        """
        x = super().forward(x, **kwargs)

        if vace_hints is not None and getattr(self, "vace_block_id", None) is not None:
            hint = vace_hints[self.vace_block_id]
            if hint.device != x.device:
                hint = hint.to(x.device)
            x = x + hint * vace_context_scale

        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings=None):
        if operation_settings is None:
            operation_settings = {}
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
        is_i2v_type=False,
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
        vace_layers=None,
        vace_in_dim=None,
        operations=None,
        device=None,
        dtype=None,
        sp_size=1,
        model_type: str | None = None,
        is_turbo_diffusion_wan_model=False,
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
            vace_layers (`int`, *optional*, defaults to None):
                Number of VACE blocks (only for model_type='vace').
                If None, defaults to num_layers // 2
            vace_in_dim (`int`, *optional*, defaults to None):
                Input dimension for VACE context (only for model_type='vace').
                If None, defaults to in_dim
            sp_size (`int`, *optional*, defaults to 1):
                Sequence parallelism size
            is_turbo_diffusion_wan_model (`bool`, *optional*, defaults to False):
                is turbo diffusion wan model
        """

        super().__init__()
        if model_type is None:
            model_type = "i2v" if is_i2v_type else "t2v"
        assert model_type in ["t2v", "i2v", "ti2v", "s2v", "vace"]
        self.model_type = model_type
        self.is_i2v_type = model_type in ["i2v", "ti2v", "s2v"] or is_i2v_type

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

        self.is_turbo_diffusion_wan_model = is_turbo_diffusion_wan_model

        if not hasattr(self, "vace_num_blocks"):
            self.vace_num_blocks = 0
            self.vace_in_dim = None
            self.vace_layers_mapping = {}

        if self.is_turbo_diffusion_wan_model:
            self.patch_embedding = nn.Linear(in_dim * patch_size[0] * patch_size[1] * patch_size[2], dim)
        else:
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
        operation_settings = {
            "operations": operations,
            "device": device,
            "dtype": dtype,
            "rms_dtype": getattr(operations, "rms_dtype", None),
        }

        self.blocks = self._create_blocks(operation_settings)

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
        self.rope_embedder = EmbedND(
            dim=d,
            theta=10000.0,
            axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
        )
        self.rope_func = "default"
        self.cached_comfy_freqs = None
        self.cached_comfy_shape = None

        # video_attention_split_steps: steps where attention should be split for multi-prompt
        # This is set externally by the generator based on experimental_args
        self.video_attention_split_steps = []

    def _create_blocks(self, operation_settings):
        return nn.ModuleList(
            [
                WanAttentionBlock(
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    block_id,
                    self.window_size,
                    self.qk_norm,
                    self.cross_attn_norm,
                    self.eps,
                    operation_settings=operation_settings,
                    is_turbo_diffusion_wan_model=self.is_turbo_diffusion_wan_model,
                )
                for block_id in range(self.num_layers)
            ]
        )

    def set_rope_function(self, rope_function: str | None):
        rope_value = rope_function or "default"
        if rope_value != self.rope_func:
            self.cached_comfy_freqs = None
            self.cached_comfy_shape = None
        self.rope_func = rope_value

    def _rope_encode(self, grid_sizes, device, dtype):
        sizes = grid_sizes[0].tolist()
        # 最后一个维度是shape是3。表示某个 token 在 (f, h, w) 三个轴上的位置索引
        f, h, w = int(sizes[0]), int(sizes[1]), int(sizes[2])
        img_ids = torch.zeros((f, h, w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(
            0, f - 1, steps=f, device=device, dtype=dtype
        ).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(
            0, h - 1, steps=h, device=device, dtype=dtype
        ).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(
            0, w - 1, steps=w, device=device, dtype=dtype
        ).reshape(1, 1, -1)
        img_ids = img_ids.reshape(1, -1, img_ids.shape[-1])
        # img_ids.shape = [1,f*h*w,3]
        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return freqs.to(dtype=dtype)

    def _get_rope_freqs(self, grid_sizes, device, dtype):
        if "comfy" in self.rope_func:
            shape_key = (tuple(grid_sizes.reshape(-1).tolist()), dtype, device, self.rope_func)
            if (
                self.cached_comfy_freqs is None
                or self.cached_comfy_shape != shape_key
                or self.cached_comfy_freqs.device != device
                or self.cached_comfy_freqs.dtype != dtype
            ):
                self.cached_comfy_freqs = self._rope_encode(grid_sizes, device, dtype)
                self.cached_comfy_shape = shape_key
            return self.cached_comfy_freqs
        return self.freqs

    def set_keep_in_fp32_modules(self):
        fp32_modules = [
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
        self._keep_in_fp32_modules = fp32_modules
        self._keep_in_fp32_params = self._find_fp32_params(["modulation"])

    def _find_fp32_params(self, keywords):
        return [name for name, _ in self.named_parameters() if any(keyword in name for keyword in keywords)]

    def _get_seq_len(self, latent_shape):
        if len(latent_shape) != 5:
            raise ValueError("latent_shape must be of length 5 (B, C, F, H, W)")
        if len(self.patch_size) < 3:
            raise ValueError("patch_size must be of length 3 (t_patch, h_patch, w_patch)")
        _, _, lat_f, lat_h, lat_w = latent_shape

        max_seq_len = (lat_f * lat_h * lat_w) // (self.patch_size[1] * self.patch_size[2])
        return int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

    def _run_blocks(self, x, cache, phase, timestep, kwargs, slg_active, slg_blocks, seq_len, **extra_kwargs):
        if cache is None:
            for block_idx, block in enumerate(self.blocks):
                if slg_active and block_idx in slg_blocks:
                    continue
                x = block(x, **kwargs)
        else:
            x = self._forward_with_cache(x, cache, phase, timestep, **kwargs)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t,
        step_iter: int,
        cache: KsanaHybridCache | None,
        phase: str,
        context: torch.Tensor,
        y=None,
        slg_config: KsanaSLGConfig | None = None,
        feta_config: KsanaFETAConfig | None = None,
        current_step_percent: float = 0.0,
        **extra_kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                Input video tensor with shape [B, C_in, F, H, W], : [bs, 16, v, h, w]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            step_iter (int):
                Current step iteration
            cache (KsanaCache):
                Cache object for caching intermediate results
            phase (str):
                'cond' or 'uncond' phase
            context (Tensor):
                Text embeddings tensor with shape [B, L, C] : [bs, 73, 4096]
            seq_len (`int`): 7200
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            slg_config (KsanaSLGConfig, *optional*):
                Skip Layer Guidance config. When provided, skips uncond inference on
                specified blocks for speedup. See KsanaSLGConfig for details.
            feta_config (KsanaFETAConfig, *optional*):
                Enhance-A-Video (FETA) config. When provided, improves temporal
                consistency by modulating attention. See KsanaFETAConfig for details.
            current_step_percent (float, *optional*, defaults to 0.0):
                Current step as percentage of total steps (used for SLG/FETA scheduling)

        Returns:
            Tensor:
                Denoised video tensor with shape [B, C_out, F, H / 8, W / 8]
        """
        latent_shape = x.shape
        seq_len = self._get_seq_len(latent_shape)

        # x [bs, 16, f, h, w], y [bs, 20, f, h, w]
        if self.is_i2v_type and y is None:
            raise ValueError("y must be provided for i2v model")
        # params
        device = self.patch_embedding.weight.device

        if y is not None:
            # x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            x = torch.cat([x, y.to(x.dtype)], dim=1)

        kt, kh, kw = self.patch_size
        bs, _, t_in, h_in, w_in = x.shape
        if not ((t_in % kt) == 0 and (h_in % kh) == 0 and (w_in % kw) == 0):
            raise RuntimeError(
                f"Input video size {(t_in, h_in, w_in)} is not divisible by patch size {self.patch_size}"
            )

        t_in, h_in, w_in = t_in // kt, h_in // kh, w_in // kw
        # grid_sizes: 支持 batch - 所有样本共享相同的 grid (假设 batch 内尺寸相同)
        grid_sizes = torch.tensor([t_in, h_in, w_in], dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)

        # embeddings
        if self.is_turbo_diffusion_wan_model:
            x = rearrange(
                x,
                "b c (t kt) (h kh) (w kw) -> b (t h w) (c kt kh kw)",
                kt=kt,
                kh=kh,
                kw=kw,
            ).contiguous()
            x = self.patch_embedding(x.float()).to(x.dtype)
        else:
            # [bs, 16, fi, hi, wi] => [bs, 5120, f, h, w]
            x = self.patch_embedding(x.float()).to(x.dtype)
            # [bs, 5120, f*h*w] => [bs, f*h*w, 5120]
            x = x.flatten(2).transpose(1, 2)

        # seq_lens: 支持 batch
        seq_lens = torch.full((bs,), x.shape[1], dtype=torch.int32, device=x.device)
        assert seq_lens.max() <= seq_len
        x = torch.cat(
            [x, x.new_zeros(x.size(0), seq_len - x.size(1), x.size(2))], dim=1
        )  # pad f*h*w to seqlen, => [bs, seqlen, 5120]

        # time embeddings
        # 取第一个 timestep，因为一个batch里的timestamp都是一样的
        timestep = t[0].item() if t.numel() > 1 else t.item()

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

        # context
        context_lens = None
        # [bs, 512, 4096] pad to => [bs, text_len:512, 4096]
        padded_context = torch.cat(
            [context, context.new_zeros(bs, self.text_len - context.size(1), context.size(2))], dim=1
        )
        # [bs, text_len, 4096] => [bs, text_len, 5120]
        context = self.text_embedding(padded_context)

        if self.sp_size > 1:
            x = torch.chunk(x, self.sp_size, dim=1)[get_rank_id()]

        # RoPE frequencies
        rope_freqs = self._get_rope_freqs(grid_sizes, device=x.device, dtype=x.dtype)
        if rope_freqs.device != device:
            rope_freqs = rope_freqs.to(device)

        num_frames = grid_sizes[0, 0].item() if grid_sizes.numel() > 0 else 1

        feta_enabled = False
        feta_weight = KsanaFETAConfig().weight
        if feta_config is not None:
            feta_enabled = feta_config.start_percent <= current_step_percent <= feta_config.end_percent
            feta_weight = feta_config.weight

        # arguments for attention blocks
        kwargs = dict(
            e=e0,  # [bs, seqlen, 6, 5120]
            seq_lens=seq_lens,  # [bs]
            grid_sizes=grid_sizes,  # [bs, 3]
            freqs=rope_freqs,  # [1024, 64]
            context=context,  # [bs, text_len:512, 5120]
            context_lens=context_lens,
            step_iter=step_iter,
            latent_shape=latent_shape,
            rope_func=self.rope_func,
            feta_enabled=feta_enabled,
            feta_weight=feta_weight,
            num_frames=num_frames,
        )

        is_uncond = phase == "uncond"
        slg_blocks = slg_config.blocks if slg_config is not None else []
        slg_active = (
            len(slg_blocks) > 0
            and is_uncond
            and slg_config is not None
            and slg_config.start_percent <= current_step_percent <= slg_config.end_percent
        )

        # x: [bs, seqlen, 5120]
        x = self._run_blocks(x, cache, phase, timestep, kwargs, slg_active, slg_blocks, seq_len, **extra_kwargs)

        # head
        # [bs, seqlen, 5120] => [bs, seqlen, 64], e:[bs, seqlen, freq_dim=>self.dim:5120]
        x = self.head(x, e)
        if self.sp_size > 1:
            x = gather_forward(x, dim=1)
        # unpatchify
        # [bs, seqlen, 64] => [bs, 16, fi, hi, wi]
        x = self.unpatchify(x, grid_sizes)

        return x

    def _forward_with_cache(self, x, cache, phase, timestep, *, step_iter, **kwargs):
        step_cache, block_cache = cache.step_cache, cache.block_cache
        e = kwargs.get("e")
        # EasyCache and TeaCache need e for cache hit validation
        if step_cache is not None and step_cache.valid_for(
            phase=phase, x=x, step_iter=step_iter, timestep=timestep, e=e
        ):
            x = step_cache(phase=phase, x=x, step_iter=step_iter, timestep=timestep)
            return x  # step cache hit, skip block cache and return

        if step_cache is not None:
            step_cache.record_input_before_update(x=x, step_iter=step_iter, timestep=timestep)

        if block_cache is not None:
            x = block_cache(phase=phase, x=x, step_iter=step_iter, timestep=timestep, blocks=self.blocks, **kwargs)
        else:
            for block in self.blocks:
                x = block(x, step_iter=step_iter, **kwargs)

        if step_cache is not None:
            step_cache.update_cache(phase=phase, x=x, step_iter=step_iter, timestep=timestep)
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


class VaceWanModel(WanModel):
    _no_split_modules = ["BaseWanVaceAttentionBlock", "VaceWanAttentionBlock"]

    def __init__(self, **kwargs):
        vace_layers = kwargs.pop("vace_layers", None)
        vace_in_dim = kwargs.pop("vace_in_dim", None)
        num_layers = kwargs.get("num_layers", 32)
        in_dim = kwargs.get("in_dim", 16)

        if vace_layers is None:
            self.vace_num_blocks = num_layers // 2
        elif isinstance(vace_layers, int):
            self.vace_num_blocks = vace_layers
        else:
            raise ValueError(f"vace_layers must be None or int, got {type(vace_layers)}")

        self.vace_in_dim = vace_in_dim if vace_in_dim is not None else in_dim
        layer_step = num_layers // self.vace_num_blocks
        self.vace_layers_mapping = {i: n for n, i in enumerate(range(0, num_layers, layer_step))}

        operations = kwargs.get("operations")
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        kwargs.setdefault("model_type", "vace")
        super().__init__(vace_layers=vace_layers, vace_in_dim=vace_in_dim, **kwargs)

        # VACE-specific modules
        operation_settings = {
            "operations": operations,
            "device": device,
            "dtype": dtype,
            "rms_dtype": getattr(operations, "rms_dtype", None),
        }

        self.vace_blocks = nn.ModuleList(
            [
                VaceWanAttentionBlock(
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    block_id=idx,
                    window_size=self.window_size,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    operation_settings=operation_settings,
                )
                for idx in range(self.vace_num_blocks)
            ]
        )
        self.vace_patch_embedding = operations.Conv3d(
            self.vace_in_dim,
            self.dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            device=device,
            dtype=torch.float32,
        )

    def _create_blocks(self, operation_settings):
        blocks = nn.ModuleList(
            [
                BaseWanVaceAttentionBlock(
                    self.dim,
                    self.ffn_dim,
                    self.num_heads,
                    block_id=i,
                    window_size=self.window_size,
                    qk_norm=self.qk_norm,
                    cross_attn_norm=self.cross_attn_norm,
                    eps=self.eps,
                    operation_settings=operation_settings,
                )
                for i in range(self.num_layers)
            ]
        )
        for i, block in enumerate(blocks):
            block.vace_block_id = self.vace_layers_mapping.get(i, None)
        return blocks

    def set_keep_in_fp32_modules(self):
        super().set_keep_in_fp32_modules()
        self._keep_in_fp32_modules.append("vace_patch_embedding")

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]

        c = [self.vace_patch_embedding(u.unsqueeze(0).to(device).float()).to(dtype) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in c])

        if c.shape[0] != batch_size:
            if batch_size % c.shape[0] == 0:
                c = c.repeat(batch_size // c.shape[0], 1, 1)
            else:
                raise ValueError(
                    f"VACE batch size mismatch: x has batch {batch_size}, "
                    f"but vace_context produces batch {c.shape[0]}."
                )

        if self.sp_size > 1:
            c = torch.chunk(c, self.sp_size, dim=1)[get_rank_id()]
        else:
            if x.shape[1] > c.shape[1]:
                c = torch.cat([c.new_zeros(batch_size, x.shape[1] - c.shape[1], c.shape[2]), c], dim=1)
            if c.shape[1] > x.shape[1]:
                c = c[:, : x.shape[1]]

        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block in self.vace_blocks:
            c_skip, c = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

    def _run_blocks(
        self,
        x,
        cache,
        phase,
        timestep,
        kwargs,
        slg_active,
        slg_blocks,
        seq_len,
        vace_context=None,
        vace_context_scale=1.0,
        **extra_kwargs,
    ):
        vace_hints = None
        if vace_context is not None:
            vace_hints = self.forward_vace(x, vace_context, seq_len, kwargs)

        if cache is None:
            for block_idx, block in enumerate(self.blocks):
                if slg_active and block_idx in slg_blocks:
                    continue
                x = block(x, vace_hints=vace_hints, vace_context_scale=vace_context_scale, **kwargs)
        else:
            x = self._forward_with_cache(x, cache, phase, timestep, **kwargs)
        return x

    def forward(
        self,
        x,
        t,
        step_iter,
        cache,
        phase,
        context,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
        slg_config=None,
        feta_config=None,
        current_step_percent=0.0,
    ):
        return super().forward(
            x,
            t,
            step_iter,
            cache,
            phase,
            context,
            y=y,
            slg_config=slg_config,
            feta_config=feta_config,
            current_step_percent=current_step_percent,
            vace_context=vace_context,
            vace_context_scale=vace_context_scale,
        )

    @time_range
    def init_weights(self):
        super().init_weights()
        nn.init.xavier_uniform_(self.vace_patch_embedding.weight.flatten(1))
        for block in self.vace_blocks:
            if hasattr(block, "before_proj"):
                nn.init.zeros_(block.before_proj.weight)
                if block.before_proj.bias is not None:
                    nn.init.zeros_(block.before_proj.bias)
            nn.init.zeros_(block.after_proj.weight)
            if block.after_proj.bias is not None:
                nn.init.zeros_(block.after_proj.bias)
