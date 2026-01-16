"""
Reference (Diffusers):
  - diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py
"""

from .base_x2x import KsanaPipeline


class KsanaQwenImageT2IPipeline(KsanaPipeline):
    def __init__(
        self,
    ):
        # self.default_args = QwenImageDefaultArgs()
        default_cfg = getattr(self.pipeline_config, "default_config", None)
        if default_cfg is not None:
            if hasattr(default_cfg, "z_dim"):
                self.vae_z_dim = default_cfg.z_dim
            if hasattr(default_cfg, "vae_stride"):
                self.vae_stride = default_cfg.vae_stride
            self.vae_scale_factor = getattr(default_cfg, "vae_scale_factor", 8)
        else:
            self.vae_scale_factor = 8
