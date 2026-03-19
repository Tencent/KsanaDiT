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

"""Latent shape 计算工具。

提供统一的 :func:`compute_latent_shape` 基础函数，
以及 :func:`create_empty_noise_latent` 辅助函数。
"""


import numpy as np
import torch


def _normalize_to_3d(value: int | list[int], name: str) -> list[int]:
    """将标量或列表统一为 ``[f, h, w]`` 三维列表。

    - 标量 ``v`` → ``[1, v, v]``（仅空间维度生效，时间维度设为 1）
    - 长度 2 → ``[1, h, w]``
    - 长度 3 → 原样返回
    """
    if isinstance(value, (int, float)):
        return [1, int(value), int(value)]
    value = list(value)
    if len(value) == 2:
        return [1, int(value[0]), int(value[1])]
    if len(value) == 3:
        return [int(v) for v in value]
    raise ValueError(f"{name} must be int or list of length 2 or 3, got {value}")


def compute_latent_shape(
    z_dim: int,
    target_f: int,
    target_h: int,
    target_w: int,
    vae_stride: list[int],
    patch_size: int | list[int],
    refer_image_shape: list[int] | None = None,
) -> tuple[int, int, int, int]:
    """根据 VAE 配置计算 latent 形状 ``(z_dim, lat_f, lat_h, lat_w)``。

    这是 video / image latent shape 计算的统一基础函数。

    Parameters
    ----------
    z_dim:
        latent 通道数。
    target_f:
        目标帧数。图像场景传 ``1``。
    target_h, target_w:
        目标像素高/宽。
    vae_stride:
        VAE 下采样步长，**必须**为 ``[stride_f, stride_h, stride_w]`` 三元素列表。
    patch_size:
        Patch 大小。可以是 ``int``（图像场景，h/w 共享）或
        ``[patch_f, patch_h, patch_w]`` 列表（视频场景）。
        标量 ``p`` 会被展开为 ``[1, p, p]``。
    refer_image_shape:
        可选，``[bs, 3, ih, iw]`` 格式的参考图尺寸。
        提供时会按图片宽高比修正 latent 空间尺寸，使 latent 面积
        ≈ ``target_h * target_w / (stride * patch)^2`` 但宽高比与图片一致。

    Returns
    -------
    tuple[int, int, int, int]
        ``(z_dim, lat_f, lat_h, lat_w)``
    """
    if len(vae_stride) != 3:
        raise ValueError(f"vae_stride must be a list of 3 elements [f, h, w], got {vae_stride}")

    patch = _normalize_to_3d(patch_size, "patch_size")

    if refer_image_shape is not None:
        if len(refer_image_shape) != 4 or refer_image_shape[1] != 3:
            raise ValueError(f"refer_image_shape must be 4D [bs, 3, h, w], got {refer_image_shape}")
        img_h, img_w = refer_image_shape[2], refer_image_shape[3]
    else:
        img_h, img_w = target_h, target_w

    lat_h = round(np.sqrt(target_w * target_h * (img_h / img_w)) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(np.sqrt(target_w * target_h * (img_w / img_h)) // vae_stride[2] // patch[2] * patch[2])
    lat_f = (target_f - 1) // vae_stride[0] + 1

    return z_dim, lat_f, lat_h, lat_w


def create_empty_noise_latent(
    latent_shape: tuple[int, int, int, int],
    batch_size: int = 1,
) -> torch.Tensor:
    """根据 ``(z_dim, lat_f, lat_h, lat_w)`` 创建全零 latent tensor。

    Parameters
    ----------
    latent_shape:
        ``(z_dim, lat_f, lat_h, lat_w)`` — 由 :func:`compute_latent_shape` 返回。
    batch_size:
        batch 维度大小，默认 1。

    Returns
    -------
    torch.Tensor
        shape ``(batch_size, z_dim, lat_f, lat_h, lat_w)``，device=cpu。
    """
    return torch.zeros(batch_size, *latent_shape, device="cpu")
