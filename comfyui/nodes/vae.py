# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import folder_paths

import ksana.nodes as nodes
from ksana.nodes import KSANA_CATEGORY_VAE, KSANA_GENERATE_OUTPUT, KSANA_VAE_ENCODE_OUTPUT, KSANA_VAE_MODEL


class KsanaVAELoaderNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}

    RETURN_TYPES = (KSANA_VAE_MODEL,)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = KSANA_CATEGORY_VAE

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        print(vae_path)
        return (nodes.KsanaNodeVAELoader.load(vae_path),)


class KsanaVAEEncodeNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "optional": {
                "vae": (KSANA_VAE_MODEL, {"tooltip": "The KsanaVAEModel used for encoding the input image."}),
                "start_image": ("IMAGE", {"tooltip": "The start image to encode."}),
                "end_image": ("IMAGE", {"tooltip": "The end image to encode."}),
                "mask": ("MASK", {"tooltip": "The mask to apply to the image."}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 81}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 2048}),
                "height": ("INT", {"default": 720, "min": 1, "max": 2048}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
        }

    RETURN_TYPES = (KSANA_VAE_ENCODE_OUTPUT,)
    RETURN_NAMES = ("latent",)
    FUNCTION = "vae_encode"
    CATEGORY = KSANA_CATEGORY_VAE

    def vae_encode(self, *args, **kwargs):
        return (nodes.vae_encode(*args, **kwargs),)


class KsanaVAEImageEncodeNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "optional": {
                "vae": (KSANA_VAE_MODEL, {"tooltip": "The KsanaVAEModel used for encoding the input image."}),
                "image": ("IMAGE", {"tooltip": "The image to encode."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 32}),
            },
        }

    RETURN_TYPES = (KSANA_VAE_ENCODE_OUTPUT,)
    RETURN_NAMES = ("latent",)
    FUNCTION = "vae_encode_image"
    CATEGORY = KSANA_CATEGORY_VAE

    def vae_encode_image(self, *args, **kwargs):
        return (nodes.vae_encode_image(*args, **kwargs),)


class KsanaVAEDecodeNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "vae": (KSANA_VAE_MODEL, {"tooltip": "The KsanaVAEModel used for encoding the input image."}),
                "latent": (KSANA_GENERATE_OUTPUT, {"tooltip": "The latent tensor to decode."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "vae_decode"
    CATEGORY = KSANA_CATEGORY_VAE

    def vae_decode(self, *args, **kwargs):
        return (nodes.vae_decode(*args, **kwargs),)
