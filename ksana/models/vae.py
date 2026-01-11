import os
from pathlib import Path

import numpy as np
import torch

from ..models.model_key import QWEN_IMAGE, WAN2_1, WAN2_2, KsanaModelKey
from ..utils.logger import log
from ..utils.profile import time_range
from ..utils.types import any_key_in_str
from .base_model import KsanaModel
from .qwen.vae import KsanaQwenImageVAE
from .wan import Wan2_1_VAE, Wan2_2_VAE


class KsanaVAE(KsanaModel):
    def __init__(self, model_path, device, vae_type=None, dtype=torch.float):
        self.device = device
        self.dtype = dtype
        if vae_type is None:
            vae_type = KsanaVAE.get_model_type_from_path(model_path)
        if vae_type in WAN2_1:
            self.model = Wan2_1_VAE(
                vae_pth=model_path,
                dtype=dtype,
                device=device,
            )
            self._key = KsanaModelKey.VAE_WAN2_1
        elif vae_type in WAN2_2:
            self.model = Wan2_2_VAE(
                vae_pth=model_path,
                dtype=dtype,
                device=device,
            )
            self._key = KsanaModelKey.VAE_WAN2_2
        elif vae_type in QWEN_IMAGE:
            self.model = KsanaQwenImageVAE(vae_path=model_path, device=device, dtype=dtype)
            self._key = KsanaModelKey.QwenImageVAE
        else:
            raise ValueError(f"vae_type {vae_type} not supported")

        self.z_dim = self.model.model.z_dim
        # Note: wan 14B: (4, 8, 8), 5B: (4, 16, 16)
        self.vae_stride = [4, 8, 8]
        self.vae_patch = [1, 2, 2]
        log.info(f"self.z_dim {self.z_dim}, vae_stride {self.vae_stride}, vae_patch {self.vae_patch}")

    def get_model_key(self) -> KsanaModelKey:
        return self._key

    @staticmethod
    def get_model_type_from_path(model_path: str):
        file_name = os.path.basename(model_path).lower()
        if (
            Path(model_path).is_file()
            and any_key_in_str(QWEN_IMAGE, file_name) is not None
            and file_name.find("hf") == -1
        ):
            return WAN2_1[0]
        if any_key_in_str(WAN2_2, file_name) is not None:
            return WAN2_2[0]
        elif any_key_in_str(WAN2_1, file_name) is not None:
            return WAN2_1[0]
        elif any_key_in_str(QWEN_IMAGE, file_name) is not None:
            return QWEN_IMAGE[0]
        else:
            raise ValueError(f"model_path {model_path} not in support list {WAN2_2 + WAN2_1 + QWEN_IMAGE}")

    @time_range("vae_decode")
    def decode(self, latents, with_end_image: bool = False):
        return self.model.decode(latents, with_end_image)

    @time_range("vae_encode")
    def encode(self, latents, with_end_image: bool = False):
        return self.model.encode(latents, with_end_image)

    def to(self, device):
        if self.device != device or self.model.device != device:
            self.model.to(device)
        self.device = device
        return self

    def get_img_mask(self, bs, lat_f, lat_h, lat_w, device, has_end_img: bool = False, vae_stride=None):
        vae_stride = vae_stride or self.vae_stride
        start = torch.ones(bs, vae_stride[0], lat_h, lat_w, device=device)
        if has_end_img:
            zeros = torch.zeros(bs, (lat_f - 1) * vae_stride[0], lat_h, lat_w, device=device)
            end = torch.ones(bs, vae_stride[0], lat_h, lat_w, device=device)
            msk = torch.concat([start, zeros, end], dim=1)
        else:
            zeros = torch.zeros(bs, (lat_f - 1) * vae_stride[0], lat_h, lat_w, device=device)
            msk = torch.concat([start, zeros], dim=1)

        msk = msk.view(bs, -1, vae_stride[0], lat_h, lat_w)
        msk = msk.transpose(1, 2)
        return msk

    def forward_encode(
        self,
        target_f: int,
        target_h: int,
        target_w: int,
        *,
        device,
        target_batch_size: int,
        start_img: torch.Tensor = None,
        end_img: torch.Tensor = None,
        mask: torch.Tensor = None,
        vae_stride: list[int] = None,
        vae_patch: list[int] = None,
    ):
        vae_stride = vae_stride or self.vae_stride
        vae_patch = vae_patch or self.vae_patch

        with_end_image = end_img is not None

        if start_img is not None:
            assert start_img.ndim == 4, f"img_batch must be 4D tensor[bs, 3, h, w], but got shape {start_img.shape}"
            assert start_img.shape[1] == 3, f"start_img must be 4D tensor[bs, 3, h, w], but got shape {start_img.shape}"
            if with_end_image:
                assert (
                    start_img.shape == end_img.shape
                ), f"start_img and end_img must have same shape, but got {start_img.shape} and {end_img.shape}"

        # img: [bs, 3, ih, iw]
        img_h, img_w = start_img.shape[2:] if start_img is not None else (target_h, target_w)
        lat_h = round(np.sqrt(target_w * target_h * (img_h / img_w)) // vae_stride[1] // vae_patch[1] * vae_patch[1])
        lat_w = round(np.sqrt(target_w * target_h * (img_w / img_h)) // vae_stride[2] // vae_patch[2] * vae_patch[2])
        lat_f = (target_f - 1) // vae_stride[0] + 1

        if start_img is None:
            return torch.zeros(target_batch_size, self.z_dim, lat_f, lat_h, lat_w, device="cpu")

        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]
        start_img_batch = torch.nn.functional.interpolate(start_img.cpu(), size=(h, w), mode="bicubic")
        if with_end_image:
            end_img_batch = torch.nn.functional.interpolate(end_img.cpu(), size=(h, w), mode="bicubic")
        # for bs in [bs, 3, h, w]
        merge_batch = []
        bs = start_img_batch.shape[0]
        for i in range(bs):
            one_start_img = start_img_batch[i]
            # one_img: [3, h, w] => [1, 3, h, w] => [3, 1, h, w]
            one_start_img = one_start_img.unsqueeze(0).transpose(0, 1)
            if with_end_image:
                one_end_img = end_img_batch[i]
                one_end_img = one_end_img.unsqueeze(0).transpose(0, 1)
                merge = torch.concat([one_start_img, torch.zeros(3, target_f - 1, h, w), one_end_img], dim=1)
            else:
                merge = torch.concat([one_start_img, torch.zeros(3, target_f - 1, h, w)], dim=1)
            merge_batch.append(merge.unsqueeze(0))

        del start_img_batch
        if with_end_image:
            del end_img_batch

        merge_batch = torch.concat(merge_batch, dim=0)
        # merge_batch [bs, 3, target_f, h, w] => y [bs, 16, lat_f, lat_h , lat_w]
        current_device = self.device
        if current_device != device:
            self.to(device)

        y = self.encode(merge_batch.to(device), with_end_image)
        self.to(current_device)
        assert y.shape == (
            bs,
            self.z_dim,
            y.shape[2],
            lat_h,
            lat_w,
        ), (
            "vae encode shape must be [bs, 16, lat_f, lat_h, lat_w], "
            f"but got {y.shape}, {[bs, self.z_dim, y.shape[2], lat_h, lat_w]}"
        )

        del merge_batch

        if mask is None:
            mask = self.get_img_mask(bs, lat_f, lat_h, lat_w, device, with_end_image, vae_stride)

        y = torch.concat([mask, y], dim=1)

        if target_batch_size > bs:
            y = y.repeat(target_batch_size // bs, 1, 1, 1, 1)

        log.info(f"image_latents shape {y.shape}, {y.device}, {y.dtype}, mask shape {mask.shape}")
        return y

    @time_range
    def forward_decode(self, latents, local_rank, device=None, with_end_image: bool = False) -> torch.Tensor:
        # TODO: support multi gpu
        if local_rank != 0:
            return
        log.info(f"vae_decode input latents shape: {latents.shape}, {latents.device}, {latents.dtype}")
        if latents.ndim != 5:
            raise RuntimeError(
                f"latents must be 5D tensor[bs, z_dim, lat_f, lat_h, lat_w], but got shape {latents.shape}"
            )
        current_device = self.device
        if current_device != device:
            self.to(device)
        torch.cuda.empty_cache()
        # TODO: Enable batch decoding when latent size is small enough to fit in memory
        decoded_latents = []
        for i in range(latents.shape[0]):
            decoded_latents.append(self.decode(latents[i : i + 1], with_end_image=with_end_image))
        latents = torch.cat(decoded_latents, dim=0)
        self.to(current_device)
        if with_end_image:
            latents = latents[:, :, 0:-1]
        log.info(f"decode output shape: {latents.shape}")
        # return [bs, ch:3, f, h, w]
        return latents
