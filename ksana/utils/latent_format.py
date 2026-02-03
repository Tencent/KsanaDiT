import torch


class Wan21LatentFormat:
    """
    WAN2.1 latent format for process_in/process_out operations.
    Compatible with ComfyUI's Wan21 latent_format.

    - process_in: Convert raw latent to normalized latent (for model input)
    - process_out: Convert normalized latent to raw latent (for VAE decode)
    """

    latent_channels = 16
    scale_factor = 1.0

    def __init__(self):
        self.latents_mean = torch.tensor(
            [
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
            ]
        ).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor(
            [
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
                1.9160,
            ]
        ).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        if len(latent.shape) == 4:
            latent = latent.unsqueeze(0)
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


# Singleton instance for reuse
_WAN21_LATENT_FORMAT = None


def get_wan21_latent_format():
    """Get singleton instance of Wan21LatentFormat."""
    global _WAN21_LATENT_FORMAT
    if _WAN21_LATENT_FORMAT is None:
        _WAN21_LATENT_FORMAT = Wan21LatentFormat()
    return _WAN21_LATENT_FORMAT
