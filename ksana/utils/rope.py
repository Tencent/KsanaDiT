from __future__ import annotations

import torch

# #################### default rope ###########################


def _split_default_rope_freqs(freqs: torch.Tensor, half_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    split_sizes = [half_dim - 2 * (half_dim // 3), half_dim // 3, half_dim // 3]
    return freqs.split(split_sizes, dim=1)  # type: ignore[return-value]


# TODO(qiannan): 为什么要pad？只有在seqlen % sp_size !=0 的时候才需要？但是这个条件成立的时候 能否all2all通信？
def _pad_freqs(original_tensor: torch.Tensor, target_len: int) -> torch.Tensor:
    seq_len, s1, s2 = original_tensor.shape
    if seq_len >= target_len:
        return original_tensor
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    return torch.cat([original_tensor, padding_tensor], dim=0)


def _build_wan_freqs_for_grid(
    freqs_split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    seq_len = frames * height * width
    return torch.cat(
        [
            freqs_split[0][:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            freqs_split[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            freqs_split[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)


@torch.amp.autocast("cuda", enabled=False)
def apply_default_rope(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    *,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> torch.Tensor:
    """Apply Wan 3D RoPE to a (possibly sequence-parallel) shard.

    Args:
        x: Tensor of shape `[B, L_local, N, D]`.
        grid_sizes: Tensor of shape `[B, 3]` containing `(F, H, W)` per sample.
        freqs: RoPE freqs, shape `[M, D//2]` (complex cis concatenated across axes).
        sp_rank: Sequence-parallel rank index.
        sp_size: Sequence-parallel world size; when `>1`, `x` is assumed to be the shard
            for global positions `[sp_rank * L_local, (sp_rank + 1) * L_local)`.

    Returns:
        Tensor with the same shape/dtype as `x`.
    """

    if x.ndim != 4:
        raise ValueError(f"expected x to have 4 dims [B, L, N, D], got {tuple(x.shape)}")

    _, local_seq_len, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    half_dim = head_dim // 2
    freqs_split = _split_default_rope_freqs(freqs, half_dim)

    outputs: list[torch.Tensor] = []
    for sample_index, (frames, height, width) in enumerate(grid_sizes.tolist()):
        frames = int(frames)
        height = int(height)
        width = int(width)
        true_seq_len = frames * height * width

        freqs_full = _build_wan_freqs_for_grid(freqs_split, frames=frames, height=height, width=width)

        if sp_size > 1:
            freqs_full = _pad_freqs(freqs_full, local_seq_len * sp_size)
            start = sp_rank * local_seq_len
            end = (sp_rank + 1) * local_seq_len
            freqs_rank = freqs_full[start:end]

            x_i = torch.view_as_complex(
                x[sample_index, :local_seq_len].to(torch.float64).reshape(local_seq_len, num_heads, -1, 2)
            )
            x_i = torch.view_as_real(x_i * freqs_rank).flatten(2)
            x_i = torch.cat([x_i, x[sample_index, local_seq_len:]])
            outputs.append(x_i)
        else:
            x_i = torch.view_as_complex(
                x[sample_index, :true_seq_len].to(torch.float64).reshape(true_seq_len, num_heads, -1, 2)
            )
            x_i = torch.view_as_real(x_i * freqs_full).flatten(2)
            x_i = torch.cat([x_i, x[sample_index, true_seq_len:]])
            outputs.append(x_i)

    return torch.stack(outputs, dim=0).type_as(x)


# #################### comfy rope ###########################


def _pad_comfy_freqs_cis(freqs_cis: torch.Tensor, target_len: int) -> torch.Tensor:
    seq_len = freqs_cis.size(1)
    if seq_len >= target_len:
        return freqs_cis
    pad_size = target_len - seq_len
    eye = torch.eye(2, dtype=freqs_cis.dtype, device=freqs_cis.device).view(1, 1, 1, 1, 2, 2)
    pad = eye.expand(
        freqs_cis.size(0),
        pad_size,
        freqs_cis.size(2),
        freqs_cis.size(3),
        2,
        2,
    )
    return torch.cat([freqs_cis, pad], dim=1)


def _comfy_rope(pos: torch.Tensor, dim: int, theta: int, device: torch.device | None = None) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    target_device = device if device is not None else pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=target_device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=target_device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32, device=pos.device)


class EmbedND(torch.nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        device = ids.device
        emb = torch.cat(
            [_comfy_rope(ids[..., i], self.axes_dim[i], self.theta, device=device) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


def apply_comfyui_rope(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    sp_rank: int = 0,
    sp_size: int = 1,
) -> torch.Tensor:
    """Apply ComfyUI-style RoPE (2x2 rotation matrices) to a (possibly SP-sharded) tensor.
    Args:
        x: `[B, L_local, N, D]`.
        freqs_cis: typically `[1, L_true, 1, D/2, 2, 2]` (broadcastable over batch/heads).
        sp_rank/sp_size: when `sp_size>1`, slice freqs for global positions
            `[sp_rank * L_local, (sp_rank + 1) * L_local)` and pad missing positions with identity.
    """

    if x.ndim != 4:
        raise ValueError(f"expected x to have 4 dims [B, L, N, D], got {tuple(x.shape)}")

    local_seq_len = x.size(1)
    if sp_size > 1:
        global_seq_len = local_seq_len * sp_size
        freqs_cis = _pad_comfy_freqs_cis(freqs_cis, global_seq_len)
        start = sp_rank * local_seq_len
        end = (sp_rank + 1) * local_seq_len
        freqs_cis = freqs_cis[:, start:end]
    else:
        freqs_cis = _pad_comfy_freqs_cis(freqs_cis, local_seq_len)
        freqs_cis = freqs_cis[:, :local_seq_len]

    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)
