"""
Reference (Diffusers):
  - diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ksana.utils import gather_forward, get_rank_id

from .attention import (
    FeedForward,
    QwenDoubleStreamAttention,
    QwenEmbedRope,
)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, operation_settings=None):
        super().__init__()
        self.time_proj = self.Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = self.TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, operation_settings=operation_settings
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb

    @staticmethod
    def _get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
    ) -> torch.Tensor:
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = scale * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    class Timesteps(nn.Module):
        def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: float):
            super().__init__()
            self.num_channels = num_channels
            self.flip_sin_to_cos = flip_sin_to_cos
            self.downscale_freq_shift = downscale_freq_shift
            self.scale = scale

        def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
            return QwenTimestepProjEmbeddings._get_timestep_embedding(
                timesteps,
                self.num_channels,
                flip_sin_to_cos=self.flip_sin_to_cos,
                downscale_freq_shift=self.downscale_freq_shift,
                scale=self.scale,
            )

    class TimestepEmbedding(nn.Module):
        def __init__(self, in_channels: int, time_embed_dim: int, operation_settings=None):
            super().__init__()
            operations = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            self.linear_1 = operations.Linear(in_channels, time_embed_dim, device=device, dtype=dtype)
            self.act = nn.SiLU()
            self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim, device=device, dtype=dtype)

        def forward(self, sample: torch.Tensor) -> torch.Tensor:
            sample = self.linear_1(sample)
            sample = self.act(sample)
            sample = self.linear_2(sample)
            return sample


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, conditioning_dim: int, eps: float = 1e-6, operation_settings=None):
        super().__init__()
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")
        self.norm = operations.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.linear = operations.Linear(conditioning_dim, dim * 2, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        emb = self.linear(nn.functional.silu(conditioning))
        scale, shift = emb.unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        operation_settings=None,
    ):
        super().__init__()
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.img_norm1 = operations.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.img_norm2 = operations.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.img_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", operation_settings=operation_settings
        )

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, device=device, dtype=dtype),
        )
        self.txt_norm1 = operations.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.txt_norm2 = operations.LayerNorm(dim, eps, elementwise_affine=False, device=device, dtype=dtype)
        self.txt_mlp = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", operation_settings=operation_settings
        )

        self.attn = QwenDoubleStreamAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qk_norm=True,
            eps=eps,
            operation_settings=operation_settings,
        )

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_out, txt_attn_out = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_out
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_out

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        operations=None,
        device=None,
        dtype=None,
        sp_size: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.sp_size = sp_size
        if operations is None:
            raise ValueError("operations parameter is required for optimized QwenImageTransformer2DModel")
        operation_settings = {
            "operations": operations,
            "device": device,
            "dtype": dtype,
            "rms_dtype": getattr(operations, "rms_dtype", None),
        }

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, operation_settings=operation_settings
        )

        rms_dtype = operation_settings.get("rms_dtype")
        self.txt_norm = operations.RMSNorm(
            joint_attention_dim, eps=1e-6, device=device, dtype=dtype, rms_dtype=rms_dtype
        )
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, device=device, dtype=dtype)

        self.img_in = operations.Linear(in_channels, self.inner_dim, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    operation_settings=operation_settings,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, eps=1e-6, operation_settings=operation_settings
        )
        self.proj_out = operations.Linear(
            self.inner_dim, patch_size * patch_size * out_channels, bias=True, device=device, dtype=dtype
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        if self.sp_size > 1:
            # TODO(qiannan): 后续这里要修改
            if hidden_states.shape[1] % self.sp_size != 0:
                raise RuntimeError(
                    f"hidden_states dimension 1: ({hidden_states.shape[1]}) "
                    f"must be divisible by sp_size ({self.sp_size})"
                )
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[get_rank_id()]
            txt_seq_len = encoder_hidden_states.shape[1]
            if txt_seq_len % self.sp_size != 0:
                pad_len = self.sp_size - (txt_seq_len % self.sp_size)
                pad_shape = list(encoder_hidden_states.shape)
                pad_shape[1] = pad_len
                pad_tensor = torch.zeros(
                    pad_shape, dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device
                )
                encoder_hidden_states = torch.cat([encoder_hidden_states, pad_tensor], dim=1)
                if encoder_hidden_states_mask is not None:
                    mask_pad_shape = list(encoder_hidden_states_mask.shape)
                    mask_pad_shape[1] = pad_len
                    mask_pad_tensor = torch.zeros(
                        mask_pad_shape, dtype=torch.bool, device=encoder_hidden_states_mask.device
                    )
                    encoder_hidden_states_mask = torch.cat([encoder_hidden_states_mask, mask_pad_tensor], dim=1)
            encoder_hidden_states = torch.chunk(encoder_hidden_states, self.sp_size, dim=1)[get_rank_id()]

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if self.sp_size > 1:
            output = gather_forward(output, dim=1)

        return output
