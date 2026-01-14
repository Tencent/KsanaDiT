"""
Reference (Diffusers):
  - diffusers/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage.py
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from ...utils.load import load_state_dict
from ...utils.lora import load_state_dict_and_merge_lora
from ..model_key import KsanaModelKey

CACHE_T = 2


def get_activation(name: str):
    if name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class QwenImageCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class QwenImageRMSNorm(nn.Module):
    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class QwenImageUpsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu"):
        super().__init__()
        self.nonlinearity = get_activation(non_linearity)
        self.norm1 = QwenImageRMSNorm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class QwenImageAttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = QwenImageRMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width).permute(0, 2, 1, 3, 4)
        return x + identity


class QwenImageMidBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)]) if upsample_mode else None

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx) if feat_cache is not None else resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache, feat_idx) if feat_cache is not None else self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_upsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.nonlinearity = get_activation(non_linearity)

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
            self.up_blocks.append(
                QwenImageUpBlock(in_dim, out_dim, num_res_blocks, dropout, upsample_mode, non_linearity)
            )

        self.norm_out = QwenImageRMSNorm(dims[-1], images=False)
        self.conv_out = QwenImageCausalConv3d(dims[-1], 3, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage(nn.Module):
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: Optional[List[float]] = None,
        latents_std: Optional[List[float]] = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImageDecoder3d(
            base_dim, z_dim, list(dim_mult), num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)
        self._feat_map = None
        self._conv_idx = None

    def clear_cache(self):
        conv_num = sum(isinstance(m, QwenImageCausalConv3d) for m in self.decoder.modules())
        self._conv_idx = [0]
        self._feat_map = [None] * conv_num

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        _, _, num_frame, height, width = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)

        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        return (out,)


DEFAULT_QWEN_IMAGE_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "latents_mean": [
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    ],
    "latents_std": [
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.916,
    ],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 16,
}


class KsanaQwenImageVAE:
    def __init__(
        self,
        vae_path: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        **_kwargs,
    ):
        path = Path(vae_path)
        config = None
        if path.is_dir():
            vae_path = os.path.join(vae_path, "vae")
            config_path = Path(vae_path) / "config.json"
            with open(config_path) as f:
                config = json.load(f)
        elif path.is_file():
            config = DEFAULT_QWEN_IMAGE_VAE_CONFIG

        self.z_dim = config.get("z_dim", 16)
        self.latents_mean = config.get("latents_mean")
        self.latents_std = config.get("latents_std")
        self.temperal_downsample = config.get("temperal_downsample", [False, True, True])
        self.vae_stride = (1, 8, 8)
        self.device = device
        self.dtype = dtype
        self.patch_size = None

        self.model = AutoencoderKLQwenImage(
            base_dim=config.get("base_dim", 96),
            z_dim=self.z_dim,
            dim_mult=tuple(config.get("dim_mult", [1, 2, 4, 4])),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=config.get("attn_scales", []),
            temperal_downsample=self.temperal_downsample,
            dropout=config.get("dropout", 0.0),
            latents_mean=self.latents_mean,
            latents_std=self.latents_std,
        )

        state_dict = load_state_dict_and_merge_lora(vae_path, device=str(device))
        load_state_dict(self.model, state_dict, strict=False)
        self.model.to(device, dtype=dtype)
        self._key = KsanaModelKey.QwenImageVAE

    def decode(self, latents: torch.Tensor, with_end_image: bool = False) -> Tuple[torch.Tensor]:
        with amp.autocast(dtype=self.dtype):
            latents_mean = torch.tensor(self.latents_mean).view(1, -1, 1, 1, 1).to(latents)
            latents_std = torch.tensor(self.latents_std).view(1, -1, 1, 1, 1).to(latents)
            latents = latents * latents_std + latents_mean
            decoded_tuple = self.model.decode(latents)
            return decoded_tuple[0].float()

    def to(self, device):
        cur = next(self.model.parameters()).device if self.model is not None else self.device
        if cur != device:
            self.model.to(device)
        self.device = device
