import torch

from ..models.model_key import KsanaModelKey
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range


@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.VAE_WAN2_1)
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.VAE_WAN2_2)
@KsanaUnitFactory.register(KsanaUnitType.ENCODER, KsanaModelKey.QwenImageVAE)
# TODO(rockcao): 区分开对image和start_img/end_img的encode方法
class KsanaVaeEncoder(KsanaRunnerUnit):
    @time_range
    def run(
        self,
        vae_model,
        *,
        start_img: torch.Tensor = None,
        end_img: torch.Tensor = None,
        mask: torch.Tensor = None,
        image: torch.Tensor = None,
        batch_size: int = None,
        **kwargs,
    ):
        if batch_size is None:
            batch_size = 1 if start_img is None else start_img.shape[0]
        if start_img is not None and batch_size % start_img.shape[0] != 0:
            raise ValueError(f"start_img batch size {start_img.shape[0]} cannot be broadcast to {batch_size}")
        log.info(
            f"vae_encode with model_key: {vae_model.model_key}, target_batch_size: {batch_size}, "
            f"start_image shape: {start_img.shape if start_img is not None else None}, "
            f"end_image shape: {end_img.shape if end_img is not None else None}, "
            f"mask shape: {mask.shape if mask is not None else None}, {kwargs}"
        )
        if image is not None and start_img is not None:
            raise ValueError("image and start_img cannot be both not None")
        if image is not None:  # only encode image
            return vae_model.forward_encode_image(
                image=image, device=kwargs.get("device"), target_batch_size=batch_size
            )
        else:
            return vae_model.forward_encode(
                start_img=start_img, end_img=end_img, mask=mask, target_batch_size=batch_size, **kwargs
            )
