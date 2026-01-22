"""
Reference (Diffusers):
  - diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
"""

import functools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ksana.utils import all_to_all, get_rank_id, get_world_size


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> torch.Tensor:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    batch, local_seq_len, head, headdim = x.shape  # pylint: disable=unused-variable
    if freqs_cis.shape[0] < local_seq_len:
        k = freqs_cis.ndim
        n = local_seq_len - freqs_cis.shape[0]
        pad_config = [0, 0] * (k - 1) + [0, n]
        freqs_cis = F.pad(freqs_cis, pad_config, value=0)
    if sp_size > 1:
        start = sp_rank * local_seq_len
        end = (sp_rank + 1) * local_seq_len
        if len(freqs_cis) < end:
            raise ValueError(
                f"The length of freqs_cis ({len(freqs_cis)}) is less than the specified end value ({end}). "
                "Ensure freqs_cis has enough elements."
            )
        freqs_rank = freqs_cis[start:end]
        if freqs_rank.shape[0] != local_seq_len:
            raise ValueError(f"freqs slice length {freqs_rank.shape[0]} != local_seq_len {local_seq_len}, ")
    else:
        freqs_rank = freqs_cis[:local_seq_len]
    freqs_rank = freqs_rank.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_rank).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

    def _rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index.float(), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        txt_seq_lens: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx).to(device)
            vid_freqs.append(video_freq)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        sp_size = get_world_size()
        if sp_size > 1:
            max_len = (max_len + sp_size - 1) // sp_size * sp_size
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...].to(device)
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


class FeedForward(nn.Module):
    class GELU(nn.Module):
        def __init__(
            self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, operation_settings=None
        ):
            super().__init__()
            operations = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            self.proj = operations.Linear(dim_in, dim_out, bias=bias, device=device, dtype=dtype)
            self.approximate = approximate

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.proj(hidden_states)
            return F.gelu(hidden_states, approximate=self.approximate)

    class GEGLU(nn.Module):
        def __init__(self, dim_in: int, dim_out: int, bias: bool = True, operation_settings=None):
            super().__init__()
            operations = operation_settings.get("operations")
            device = operation_settings.get("device")
            dtype = operation_settings.get("dtype")
            self.proj = operations.Linear(dim_in, dim_out * 2, bias=bias, device=device, dtype=dtype)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.proj(hidden_states)
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            return hidden_states * F.gelu(gate)

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        operation_settings=None,
    ):
        super().__init__()
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn: nn.Module = self.GELU(
                dim, inner_dim, approximate="none", bias=bias, operation_settings=operation_settings
            )
        elif activation_fn == "gelu-approximate":
            act_fn = self.GELU(dim, inner_dim, approximate="tanh", bias=bias, operation_settings=operation_settings)
        elif activation_fn == "geglu":
            act_fn = self.GEGLU(dim, inner_dim, bias=bias, operation_settings=operation_settings)
        else:
            raise ValueError(
                f"Unsupported activation_fn={activation_fn!r} (expected 'gelu-approximate' for Qwen-Image)"
            )

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, bias=bias, device=device, dtype=dtype))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
        operation_settings=None,
    ):
        super().__init__()
        operations = operation_settings.get("operations")
        device = operation_settings.get("device")
        dtype = operation_settings.get("dtype")

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_k = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_v = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_out = nn.ModuleList(
            [operations.Linear(dim, dim, bias=True, device=device, dtype=dtype), nn.Dropout(0.0)]
        )

        self.add_q_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.add_k_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.add_v_proj = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)
        self.to_add_out = operations.Linear(dim, dim, bias=True, device=device, dtype=dtype)

        if qk_norm:
            rms_dtype = operation_settings.get("rms_dtype")
            self.norm_q = operations.RMSNorm(head_dim, eps=eps, device=device, dtype=dtype, rms_dtype=rms_dtype)
            self.norm_k = operations.RMSNorm(head_dim, eps=eps, device=device, dtype=dtype, rms_dtype=rms_dtype)
            self.norm_added_q = operations.RMSNorm(head_dim, eps=eps, device=device, dtype=dtype, rms_dtype=rms_dtype)
            self.norm_added_k = operations.RMSNorm(head_dim, eps=eps, device=device, dtype=dtype, rms_dtype=rms_dtype)
        else:
            self.norm_q = self.norm_k = self.norm_added_q = self.norm_added_k = nn.Identity()
        self.sp_rank = get_rank_id()
        self.sp_size = get_world_size()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states.shape = [batch, seqlen, hiddim]
        seq_txt = encoder_hidden_states.shape[1]

        # q、k、v [batch, seqlen, head, headdim]
        img_q = self.to_q(hidden_states).unflatten(-1, (self.num_heads, -1))
        img_k = self.to_k(hidden_states).unflatten(-1, (self.num_heads, -1))
        img_v = self.to_v(hidden_states).unflatten(-1, (self.num_heads, -1))

        txt_q = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.num_heads, -1))
        txt_k = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.num_heads, -1))
        txt_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.num_heads, -1))

        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, self.sp_rank, self.sp_size)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, self.sp_rank, self.sp_size)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, self.sp_rank, self.sp_size)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, self.sp_rank, self.sp_size)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)

        # [batch, num_heads, seq_len, head_dim]
        joint_q = joint_q.transpose(1, 2)
        joint_k = joint_k.transpose(1, 2)
        joint_v = joint_v.transpose(1, 2)

        if self.sp_size > 1:
            # scatter head dimension, gather sequence dimension
            # attention 部分需要完整的sequence
            joint_q = all_to_all(joint_q, scatter_dim=1, gather_dim=2)
            joint_k = all_to_all(joint_k, scatter_dim=1, gather_dim=2)
            joint_v = all_to_all(joint_v, scatter_dim=1, gather_dim=2)

        attn_mask = None
        if encoder_hidden_states_mask is not None:
            bsz = encoder_hidden_states_mask.shape[0]
            seq_img = hidden_states.shape[1] * self.sp_size

            txt_valid = encoder_hidden_states_mask.to(dtype=torch.bool, device=hidden_states.device)
            img_valid = torch.ones((bsz, seq_img), dtype=torch.bool, device=hidden_states.device)
            key_valid = torch.cat([txt_valid, img_valid], dim=1)  # [B, S_total]

            neg_inf = torch.finfo(joint_q.dtype).min
            attn_mask = torch.where(
                key_valid, torch.zeros((), device=hidden_states.device, dtype=joint_q.dtype), neg_inf
            )
            attn_mask = attn_mask[:, None, None, :]

        joint_out = F.scaled_dot_product_attention(
            joint_q, joint_k, joint_v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        if self.sp_size > 1:
            # scatter sequence dimension, gather head dimension
            joint_out = all_to_all(joint_out, scatter_dim=2, gather_dim=1)

        joint_out = joint_out.transpose(1, 2).flatten(2)

        txt_out = joint_out[:, :seq_txt, :]
        img_out = joint_out[:, seq_txt:, :]

        img_out = self.to_out[0](img_out)
        img_out = self.to_out[1](img_out)
        txt_out = self.to_add_out(txt_out)

        return img_out, txt_out
