import torch

from ..models.model_key import KsanaModelKey
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range


@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.VAE_WAN2_1)
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.VAE_WAN2_2)
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.QwenImageVAE)
class KsanaVaeEncoder(KsanaRunnerUnit):
    @time_range
    def run(
        self,
        vae_model,
        *,
        start_img: torch.Tensor = None,
        end_img: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        target_batch_size = 1 if start_img is None else start_img.shape[0]
        log.info(
            f"vae_encode with model_key: {vae_model.model_key}, target_batch_size: {target_batch_size}, "
            f"start_image shape: {start_img.shape if start_img is not None else None}, "
            f"end_image shape: {end_img.shape if end_img is not None else None}, "
            f"mask shape: {mask.shape if mask is not None else None}, {kwargs}"
        )
        return vae_model.forward_encode(
            start_img=start_img, end_img=end_img, mask=mask, target_batch_size=target_batch_size, **kwargs
        )
