# Copyright 2026 Tencent
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


import torch

from kdit import get_engine
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType
from kdit.tensor import TensorKey
from kdit.utils import log

from ..output_types import EmptyLatentOutput
from ..types import KDIT_CATEGORY_UTILS, KDIT_VAE_MODEL, LATENT_OUTPUT


class EmptyLatentNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "optional": {
                "vae": (KDIT_VAE_MODEL, {"tooltip": "The kDiTVAEModel used for encoding the input image."}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 1000}),
                "width": ("INT", {"default": 1280, "min": 1, "max": 2048}),
                "height": ("INT", {"default": 720, "min": 1, "max": 2048}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
        }

    RETURN_TYPES = (LATENT_OUTPUT,)
    RETURN_NAMES = ("base_latent",)
    FUNCTION = "run_func"
    CATEGORY = KDIT_CATEGORY_UTILS

    def run_func(
        self,
        vae=None,
        num_frames=None,
        width=None,
        height=None,
        batch_size=None,
    ):
        if vae is None:
            if width is None or height is None:
                raise ValueError("width/height required if vae is None")
            # TODO: magic number
            if num_frames == 1:
                latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=torch.device("cpu"))
            else:
                latent = torch.zeros(
                    [batch_size, 16, ((num_frames - 1) // 4) + 1, height // 8, width // 8], device=torch.device("cpu")
                )
            kdit_engine = get_engine()
            kdit_engine.put_tensors(**{TensorKey.LATENTS: latent})
            return (EmptyLatentOutput(samples=TensorKey.LATENTS),)

        kdit_engine = get_engine()
        log.info(f"encoder vae: {vae}")
        context = NodeContext(
            metadata={
                "target_f": num_frames,
                "target_h": height,
                "target_w": width,
                "batch_size": batch_size,
            }
        )
        with kdit_engine.tensor_scope(keep=[TensorKey.LATENTS]):
            kdit_engine.run_infer_node(InferNodeType.VAE_ENCODE_SPATIAL, vae, context)
            kdit_engine.rename_tensor(TensorKey.BASE_LATENT, TensorKey.LATENTS)

        return (EmptyLatentOutput(samples=TensorKey.LATENTS),)
