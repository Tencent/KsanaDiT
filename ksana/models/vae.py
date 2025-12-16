import torch
from .wan import Wan2_1_VAE, Wan2_2_VAE
from .base_model import KsanaModel
from ..utils.utils import any_key_in_str
from ..models.model_key import KsanaModelKey, WAN2_2, WAN2_1


class KsanaVAE(KsanaModel):
    def __init__(self, model_path, device, vae_type=None, dtype=torch.float):
        self.device = device
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
        else:
            raise ValueError(f"model_name {self.model_name} not supported")

        self.z_dim = self.model.model.z_dim

    def get_model_key(self) -> KsanaModelKey:
        return self._key

    @staticmethod
    def get_model_type_from_path(model_path: str):
        model_path = model_path.lower()
        if any_key_in_str(WAN2_2, model_path):
            return WAN2_2[0]
        elif any_key_in_str(WAN2_1, model_path):
            return WAN2_1[0]
        else:
            raise ValueError(f"model_path {model_path} not in support list {WAN2_2 + WAN2_1}")

    def decode(self, latents):
        return self.model.decode(latents)

    def encode(self, latents):
        return self.model.encode(latents)

    def to(self, device):
        if self.device != device or self.model.device != device:
            self.model.to(device)
        self.device = device
