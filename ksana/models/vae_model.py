from abc import abstractmethod

import numpy as np
import torch

from ..models.model_key import KsanaModelKey
from ..utils.logger import log
from ..utils.profile import time_range
from .base_model import KsanaModel
from .qwen.vae import KsanaQwenImageVAE
from .wan import Wan2_1_VAE, Wan2_2_VAE


class KsanaVAEModel(KsanaModel):
    def __init__(self, model_key: KsanaModelKey, default_settings, device, dtype=torch.float32):
        super().__init__(model_key, default_settings)
        self.device = device
        self.dtype = dtype

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
        vae_stride = vae_stride or self.vae_stride_size
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

    def create_video_latent_shape(
        self, target_f: int, target_h: int, target_w: int, img_shape: list[int] = None, vae_stride=None, vae_patch=None
    ):
        vae_stride_size = vae_stride or self.vae_stride_size
        vae_patch_size = vae_patch or self.vae_patch_size
        if img_shape is not None and (len(img_shape) != 4 or img_shape[1] != 3):
            raise ValueError(f"video img_shape must be 4D tensor[bs, 3, h, w], but got shape {img_shape}")

        # img: [bs, 3, ih, iw]
        img_h, img_w = img_shape[2:] if img_shape is not None else (target_h, target_w)
        lat_h = round(
            np.sqrt(target_w * target_h * (img_h / img_w))
            // vae_stride_size[1]
            // vae_patch_size[1]
            * vae_patch_size[1]
        )
        lat_w = round(
            np.sqrt(target_w * target_h * (img_w / img_h))
            // vae_stride_size[2]
            // vae_patch_size[2]
            * vae_patch_size[2]
        )
        lat_f = (target_f - 1) // vae_stride_size[0] + 1

        if img_shape is not None:
            if len(img_shape) != 4:
                raise ValueError(f"img_shape must be 4D but got {img_shape}")
            img_h, img_w = img_shape[2:]
        else:
            img_h, img_w = target_h, target_w

        # img: [bs, 3, ih, iw]
        lat_h = round(
            np.sqrt(target_w * target_h * (img_h / img_w)) // vae_stride[1] // vae_patch_size[1] * vae_patch_size[1]
        )
        lat_w = round(
            np.sqrt(target_w * target_h * (img_w / img_h)) // vae_stride[2] // vae_patch_size[2] * vae_patch_size[2]
        )
        lat_f = (target_f - 1) // vae_stride[0] + 1
        return self.z_dim, lat_f, lat_h, lat_w

    def forward_encode_image(
        self,
        *,
        image: torch.Tensor = None,
        device,
        target_batch_size: int = 1,
    ):
        if image is None:
            return None
        # Ensure frames are in (N, C, H, W) format
        if image.ndim == 4 and image.shape[-1] == 3:
            # (N, H, W, C) -> (N, C, H, W)
            image = image.permute(0, 3, 1, 2)
        image = image.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        image = image * 2.0 - 1.0  # bs, f, c, h, w -> bs, c, f, h, w
        current_device = self.device
        if current_device != device:
            self.to(device)
        y = self.encode(image, with_end_image=False)
        self.to(current_device)
        y = y.repeat(target_batch_size, 1, 1, 1, 1)
        log.info(f"image_latents shape {y.shape}, {y.device}, {y.dtype}")
        return y

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
        vae_stride = vae_stride or self.vae_stride_size
        vae_patch_size = vae_patch or self.vae_patch_size
        img_shape = None if start_img is None else start_img.shape
        if start_img is not None and end_img is not None and start_img.shape != end_img.shape:
            raise ValueError(
                f"start_img and end_img must have same shape, but got {start_img.shape} and {end_img.shape}"
            )

        z_dim, lat_f, lat_h, lat_w = self.create_latent_shape(
            target_f=target_f,
            target_h=target_h,
            target_w=target_w,
            img_shape=img_shape,
            vae_stride=vae_stride,
            vae_patch=vae_patch_size,
        )

        if start_img is None:
            return torch.zeros(target_batch_size, z_dim, lat_f, lat_h, lat_w, device="cpu")

        with_end_image = end_img is not None

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

    @abstractmethod
    def load(self, model_path: str, *, device: torch.device, dtype=torch.float, shard_fn=None):
        pass

    @abstractmethod
    def create_latent_shape(
        self,
        *,
        target_h: int,
        target_w: int,
        target_f: int = None,
        img_shape: list[int] = None,
        vae_stride=None,
        vae_patch=None,
    ) -> tuple[int]:
        # should return [z_dim, lat_f, lat_h, lat_w]
        pass


class KsanaWanVAEModel(KsanaVAEModel):
    def load(self, model_path: str, shard_fn=None):
        self.dtype = torch.bfloat16
        if self.model_key is KsanaModelKey.VAE_WAN2_1:
            self.model = Wan2_1_VAE(vae_pth=model_path, dtype=self.dtype, device=self.device)
        elif self.model_key is KsanaModelKey.VAE_WAN2_2:
            self.model = Wan2_2_VAE(vae_pth=model_path, dtype=self.dtype, device=self.device)
        else:
            raise ValueError(f"vae model {self.model_key} not supported")
        self.z_dim = self.model.model.z_dim
        self.vae_stride_size = getattr(self.default_settings.vae, "stride", (4, 8, 8))
        self.vae_patch_size = getattr(self.default_settings.diffusion, "patch_size", [1, 2, 2])
        log.info(f"z_dim {self.z_dim}, vae_stride {self.vae_stride_size}, vae_patch_size {self.vae_patch_size}")

    def create_latent_shape(
        self,
        *,
        target_h: int,
        target_w: int,
        target_f: int = None,
        img_shape: list[int] = None,
        vae_stride=None,
        vae_patch=None,
    ):
        return self.create_video_latent_shape(target_f, target_h, target_w, img_shape, vae_stride, vae_patch)


class KsanaQwenVAEModel(KsanaVAEModel):
    def load(self, model_path: str, shard_fn=None):
        if self.model_key != KsanaModelKey.QwenImageVAE:
            raise ValueError(f"vae model {self.model_key} should be QwenImageVAE")
        self.dtype = torch.bfloat16
        self.model = KsanaQwenImageVAE(
            vae_path=model_path, device=self.device, dtype=self.dtype, default_settings=self.default_settings.vae
        )

        self.z_dim = self.model.z_dim
        self.vae_stride_size = getattr(self.default_settings.vae, "stride", (4, 8, 8))
        self.vae_patch_size = getattr(self.default_settings.diffusion, "patch_size", 2)
        self.vae_scale_factor = getattr(self.default_settings.vae, "scale_factor", 8)
        log.info(
            f"z_dim {self.z_dim}, vae_stride {self.vae_stride_size}, "
            f"vae_patch_size {self.vae_patch_size}, vae_scale_factor {self.vae_scale_factor}"
        )

    def create_latent_shape(
        self,
        *,
        target_h: int,
        target_w: int,
        target_f: int = None,
        img_shape: list[int] = None,
        vae_stride=None,
        vae_patch=None,
    ):
        if img_shape is not None:
            raise ValueError("qwen image not support edit yet")
        latent_h = target_h // self.vae_scale_factor // self.vae_patch_size * self.vae_patch_size
        latent_w = target_w // self.vae_scale_factor // self.vae_patch_size * self.vae_patch_size
        return self.z_dim, 1, latent_h, latent_w
