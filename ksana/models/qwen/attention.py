"""
Reference (Diffusers):
  - diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from ksana.accelerator import platform
from ksana.operations.fuse_qkv import QKVProjectionMixin
from ksana.utils import all_to_all, get_rank_id, get_world_size
from ksana.utils.distribute import all_gather


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
    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
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
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: int | None = None,
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


class QwenDoubleStreamAttention(nn.Module, QKVProjectionMixin):
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

        # Image stream
        self._setup_qkv_projection(
            dim=dim,
            operations=operations,
            device=device,
            dtype=dtype,
            bias=True,
            fused_name="to_qkv",
            separate_names=("to_q", "to_k", "to_v"),
            prefix="img_",
        )
        # Text stream
        self._setup_qkv_projection(
            dim=dim,
            operations=operations,
            device=device,
            dtype=dtype,
            bias=True,
            fused_name="add_qkv_proj",
            separate_names=("add_q_proj", "add_k_proj", "add_v_proj"),
            prefix="txt_",
        )

        self.to_out = nn.ModuleList(
            [operations.Linear(dim, dim, bias=True, device=device, dtype=dtype), nn.Dropout(0.0)]
        )
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
        self.attention = operations.Attn(
            num_heads=num_heads,
            head_size=head_dim,
            causal=False,
        )

    def _gather_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.sp_size == 1:
            return q, k, v
        q = all_to_all(q, scatter_dim=2, gather_dim=1)
        k = all_to_all(k, scatter_dim=2, gather_dim=1)
        v = all_to_all(v, scatter_dim=2, gather_dim=1)
        return q, k, v

    def _gather_attn_output(self, x):
        if self.sp_size == 1:
            return x
        return all_to_all(x, scatter_dim=1, gather_dim=2)

    def _gather_text_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.sp_size == 1:
            return x
        # NPU's HCCL requires contiguous tensors for all_gather operations,
        # while GPU's NCCL handles non-contiguous tensors internally.
        if platform.is_npu():
            x = x.contiguous()
        gathered = all_gather(x)
        return torch.cat(gathered, dim=2)

    def _get_text_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if self.sp_size == 1:
            return q, k, v
        heads_per_rank = self.num_heads // self.sp_size
        start_head = self.sp_rank * heads_per_rank
        end_head = (self.sp_rank + 1) * heads_per_rank

        q = q[:, :, start_head:end_head, :]
        k = k[:, :, start_head:end_head, :]
        v = v[:, :, start_head:end_head, :]
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states.shape = [batch, seqlen, hiddim]
        seq_txt = encoder_hidden_states.shape[1]

        img_q, img_k, img_v = self.compute_qkv(hidden_states, prefix="img_")
        img_q = img_q.unflatten(-1, (self.num_heads, -1))
        img_k = img_k.unflatten(-1, (self.num_heads, -1))
        img_v = img_v.unflatten(-1, (self.num_heads, -1))

        txt_q, txt_k, txt_v = self.compute_qkv(encoder_hidden_states, prefix="txt_")
        txt_q = txt_q.unflatten(-1, (self.num_heads, -1))
        txt_k = txt_k.unflatten(-1, (self.num_heads, -1))
        txt_v = txt_v.unflatten(-1, (self.num_heads, -1))

        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, self.sp_rank, self.sp_size)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, self.sp_rank, self.sp_size)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, sp_rank=0, sp_size=1)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, sp_rank=0, sp_size=1)

        img_q, img_k, img_v = self._gather_qkv(img_q, img_k, img_v)
        txt_q, txt_k, txt_v = self._get_text_qkv(txt_q, txt_k, txt_v)

        joint_q = torch.cat([txt_q, img_q], dim=1).contiguous()
        joint_k = torch.cat([txt_k, img_k], dim=1).contiguous()
        joint_v = torch.cat([txt_v, img_v], dim=1).contiguous()

        joint_out = self.attention(joint_q, joint_k, joint_v)

        txt_out = joint_out[:, :seq_txt, :]
        img_out = joint_out[:, seq_txt:, :]

        txt_out = self._gather_text_output(txt_out).flatten(2)
        img_out = self._gather_attn_output(img_out).flatten(2)

        img_out = self.to_out[0](img_out)
        img_out = self.to_out[1](img_out)
        txt_out = self.to_add_out(txt_out)

        return img_out, txt_out
