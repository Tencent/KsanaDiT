import os
from abc import ABC

import torch
from .wan import Wan2_1_VAE, Wan2_2_VAE


class KsanaVAE(ABC):
    def __init__(self, vae_type, default_pipeline_config, checkpoint_dir, device, dtype=torch.float):
        default_pipeline_config = default_pipeline_config
        self.device = device
        if vae_type == "wan2_1":
            self.model = Wan2_1_VAE(
                vae_pth=os.path.join(checkpoint_dir, default_pipeline_config.vae_checkpoint),
                dtype=dtype,
                device=device,
            )
        elif vae_type == "vae_2.2":
            self.model = Wan2_2_VAE(
                vae_pth=os.path.join(checkpoint_dir, default_pipeline_config.vae_checkpoint),
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError(f"model_name {self.model_name} not supported")

        self.z_dim = self.model.model.z_dim
        self.vae_stride = default_pipeline_config.vae_stride
        self.patch_size = default_pipeline_config.patch_size

    def decode(self, latents):
        return self.model.decode(latents)

    def to(self, device):
        if self.device != device or self.model.device != device:
            self.model.to(device)
        self.device = device
