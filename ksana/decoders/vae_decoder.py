from ..models.model_key import KsanaModelKey
from ..units import KsanaRunnerUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range


@KsanaUnitFactory.register(
    KsanaUnitType.DECODER, [KsanaModelKey.VAE_WAN2_2, KsanaModelKey.VAE_WAN2_1, KsanaModelKey.QwenImageVAE]
)
class KsanaVaeDecoder(KsanaRunnerUnit):
    @time_range
    def run(
        self,
        vae_model,
        *,
        latents,
        local_rank,
        device=None,
        offload_model=False,  # Note: add offload choice to encoder too?
        offload_device=None,
        with_end_image: bool = False,
    ):
        if local_rank != 0:
            return None
        if vae_model.device != device:
            vae_model.to(device)

        if self.model_key == KsanaModelKey.QwenImageVAE:
            # TODO(qian): maybe cloud not need if else, make it common and simpler
            latents = latents.to(vae_model.dtype)
            outputs = vae_model.decode(latents)
            # [B, C, T, H, W] -> [B, C, H, W]
            if outputs.dim() == 5:
                outputs = outputs[:, :, 0]
            outputs = (outputs + 1) / 2
            outputs = outputs.clamp(0, 1)
        else:
            outputs = vae_model.forward_decode(
                latents=latents, local_rank=local_rank, device=device, with_end_image=with_end_image
            )

        if offload_model and offload_device is not None:
            vae_model.to(offload_device)

        log.info(f"decoder output shape: {outputs.shape}")
        return outputs
