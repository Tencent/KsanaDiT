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

"""
Reference (Diffusers):
  - diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
"""

import math
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        zero_cond_t: bool = False,
        operation_settings=None,
    ):
        super().__init__()
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.zero_cond_t = zero_cond_t

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

    def _modulate(
        self, x: torch.Tensor, mod_params: torch.Tensor, index: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply modulation to input tensor, with optional per-token index selection for zero_cond_t."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)

        if index is not None:
            # mod_params batch dim is 2*actual_batch (from [real_timestep, zero_timestep])
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            shift_result = torch.where(index_expanded == 0, shift_0.unsqueeze(1), shift_1.unsqueeze(1))
            scale_result = torch.where(index_expanded == 0, scale_0.unsqueeze(1), scale_1.unsqueeze(1))
            gate_result = torch.where(index_expanded == 0, gate_0.unsqueeze(1), gate_1.unsqueeze(1))
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        modulate_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)

        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_out, txt_attn_out = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_out
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_out

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
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
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
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
        self.zero_cond_t = zero_cond_t
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
                    zero_cond_t=zero_cond_t,
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

    def _process_ref_latents(
        self,
        hidden_states: torch.Tensor,
        ref_latents: list[torch.Tensor],
    ) -> torch.Tensor:
        for ref_packed in ref_latents:
            ref_projected = self.img_in(ref_packed)

            hidden_states = torch.cat([hidden_states, ref_projected], dim=1)

        return hidden_states

    @staticmethod
    def _compute_text_seq_len_from_mask(
        encoder_hidden_states_mask: torch.Tensor,
    ) -> list[int]:
        """Compute actual text sequence lengths from attention mask (sum of True/1 values per sample)."""
        return encoder_hidden_states_mask.sum(dim=1).tolist()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_shapes: list[list[tuple[int, int, int]]] | None = None,
        txt_seq_lens: list[int] | None = None,
        ref_latents: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = self.img_in(hidden_states)
        original_seq_len = hidden_states.shape[1]

        if ref_latents is not None and len(ref_latents) > 0:
            hidden_states = self._process_ref_latents(hidden_states, ref_latents)

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # --- zero_cond_t: dual-timestep conditioning ---
        # For Qwen-Image-Edit-2511: noise latent uses real timestep, ref latents use zero timestep.
        modulate_index = None
        if self.zero_cond_t:
            # Concatenate [real_timestep, zero_timestep] for dual modulation
            timestep = torch.cat([timestep, timestep * 0], dim=0)

            # Build modulate_index: [batch, img_seq_len]
            # For each sample: first image (noise latent) gets index 0 (real timestep),
            # subsequent images (ref latents) get index 1 (zero timestep).
            modulate_index_list = []
            for b in range(batch_size):
                sample_shapes = img_shapes[b]
                indices = []
                for i, shape in enumerate(sample_shapes):
                    seq_len = prod(shape)
                    idx_val = 0 if i == 0 else 1
                    indices.extend([idx_val] * seq_len)
                modulate_index_list.append(indices)

            modulate_index = torch.tensor(modulate_index_list, dtype=torch.long, device=hidden_states.device)

        temb = self.time_text_embed(timestep, hidden_states)

        # Compute txt_seq_lens from mask if not provided
        # NOTE(qiannan): encoder_hidden_states_mask 仅用于此处推导 txt_seq_lens，
        # 未用于 attention mask。因为当前的attention暂时不支持mask。
        if txt_seq_lens is None and encoder_hidden_states_mask is not None:
            txt_seq_lens = self._compute_text_seq_len_from_mask(encoder_hidden_states_mask)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        img_seq_len = hidden_states.shape[1]
        pad_len = 0
        if self.sp_size > 1:
            # Note: qwen-image 不需要对 text 在 sequence 维度切分，只对 hidden state进行切分
            remainder = hidden_states.shape[1] % self.sp_size
            if remainder != 0:
                pad_len = self.sp_size - remainder
                hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len, 0, 0))
                if modulate_index is not None:
                    modulate_index = F.pad(modulate_index, (0, pad_len), value=0)
            hidden_states = torch.chunk(hidden_states, self.sp_size, dim=1)[get_rank_id()]
            if modulate_index is not None:
                modulate_index = torch.chunk(modulate_index, self.sp_size, dim=1)[get_rank_id()]

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                modulate_index=modulate_index,
            )

        # For norm_out, use the real-timestep temb (first half if zero_cond_t)
        if self.zero_cond_t:
            norm_temb = torch.chunk(temb, 2, dim=0)[0]
        else:
            norm_temb = temb
        hidden_states = self.norm_out(hidden_states, norm_temb)
        output = self.proj_out(hidden_states)

        if self.sp_size > 1:
            output = gather_forward(output, dim=1)
            if pad_len > 0:
                output = output[:, :img_seq_len, :]

        # Note: Edit 模式下截断 output，只返回 noise latent 部分，去除参考图像部分
        if ref_latents is not None and len(ref_latents) > 0:
            output = output[:, :original_seq_len, :]

        return output
