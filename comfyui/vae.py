import folder_paths
import torch
from ksana import get_generator
from ksana.utils import get_gpu_count, log
from ksana.config import KsanaDistributedConfig


class KsanaVAELoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}

    RETURN_TYPES = ("KSANAVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "ksana/vae"

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        print(vae_path)
        num_gpus = get_gpu_count()
        ksana_generator = get_generator(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))

        ksana_model = ksana_generator.load_vae_model(model_path=vae_path, allow_exist=True)
        return (ksana_model,)


class KsanaVAEEncodeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("KSANAVAE", {"tooltip": "The KsanaVAE model used for encoding the input image."}),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "The start image to encode."}),
                "end_image": ("IMAGE", {"tooltip": "The end image to encode."}),
                "mask": ("MASK", {"tooltip": "The mask to apply to the image."}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 81}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 1280}),
                "height": ("INT", {"default": 720, "min": 1, "max": 720}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "vae_encode"
    CATEGORY = "ksana/vae"

    def vae_encode(
        self,
        vae,
        start_image=None,
        end_image=None,
        mask=None,
        num_frames=None,
        width=None,
        height=None,
        batch_size=None,
    ):
        # TODO: support batch_size, need support batch size per one prompt
        ksana_generator = get_generator()
        print(f"encoder vae: {vae}")
        if isinstance(start_image, torch.Tensor) and start_image.ndim == 3:
            start_image = start_image.unsqueeze(0)
            print(f"start_image{start_image.shape}, {start_image.device}")
        if isinstance(end_image, torch.Tensor) and end_image.ndim == 3:
            end_image = end_image.unsqueeze(0)
            print(f"end_image{end_image.shape}, {end_image.device}")
        CHANNELS = 3
        if start_image is not None and start_image.shape[3] == CHANNELS:
            start_image = start_image.permute(0, 3, 1, 2)
        if end_image is not None and end_image.shape[3] == CHANNELS:
            end_image = end_image.permute(0, 3, 1, 2)

        def preprocess_image(image):
            if image is None:
                return image
            return image.sub(0.5).div(0.5)

        start_image = preprocess_image(start_image)
        end_image = preprocess_image(end_image)

        with_end_image = end_image is not None

        latents = ksana_generator.forward_vae_encode(
            vae_key=vae,
            frame_num=num_frames,
            width=width,
            height=height,
            start_image=start_image,
            end_image=end_image,
            mask=mask,
        )
        return ({"samples": latents, "with_end_image": with_end_image},)


class KsanaVAEDecodeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("KSANAVAE", {"tooltip": "The KsanaVAE model used for encoding the input image."}),
                "latent": ("LATENT", {"tooltip": "The latent tensor to decode."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "vae_decode"
    CATEGORY = "ksana/vae"

    def comfy_process_output(self, image):
        return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)

    def vae_decode(self, vae, latent):
        latents = latent["samples"]
        with_end_image = latent.get("with_end_image", False)
        ksana_generator = get_generator()
        print(f"latent{latents.shape}, {latents.device}")
        if isinstance(latents, torch.Tensor) and latents.ndim == 4:
            latents = latents.unsqueeze(0)
        images = ksana_generator.forward_vae_decode(
            vae_key=vae,
            latents=latents,
            with_end_image=with_end_image,
        )
        images = images.cpu().permute(0, 2, 3, 4, 1)
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        log.info(f"images {images.shape}, {images.device}")
        return (self.comfy_process_output(images),)
