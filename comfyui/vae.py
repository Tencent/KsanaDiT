import folder_paths
from ksana import get_generator
from ksana.utils import get_gpu_count
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

        ksana_model = ksana_generator.load_vae_model(
            model_path=vae_path,
        )
        return ksana_model


class KsanaVAEEncodeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("KSANAVAE",),
            },
            "optional": {
                "width": ("INT", {"default": 1280, "min": 1, "max": 1280}),
                "height": ("INT", {"default": 704, "min": 1, "max": 704}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 81}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "vae_encode"
    CATEGORY = "ksana"

    def vae_encode(self, vae, width, height, num_frames):

        ksana_generator = get_generator()
        latents = ksana_generator.vae_encoder(
            vae=vae,
            width=width,
            height=height,
            num_frames=num_frames,
        )
        return latents
