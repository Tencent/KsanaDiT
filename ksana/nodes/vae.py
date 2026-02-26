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

import torch
from ksana import get_engine
from ksana.config import KsanaDistributedConfig
from ksana.utils import get_gpu_count, log

from .output_types import KsanaNodeVAEEncodeOutput


class KsanaNodeVAELoader:
    LOADED_MODEL = None

    @classmethod
    def load(cls, vae_path):
        num_gpus = get_gpu_count()
        ksana_engine = get_engine(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            ksana_engine.clear_models(cls.LOADED_MODEL)
        cls.LOADED_MODEL = ksana_engine.load_vae_model(model_path=vae_path)
        return cls.LOADED_MODEL


def vae_encode(
    vae=None,
    start_image=None,
    end_image=None,
    mask=None,
    num_frames=None,
    width=None,
    height=None,
    batch_size=None,
):
    if vae is None:
        if width is None or height is None:
            raise ValueError("width/height required if vae is None")
        if num_frames == 1:
            latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=torch.device("cpu"))
        else:
            latent = torch.zeros(
                [batch_size, 16, ((num_frames - 1) // 4) + 1, height // 8, width // 8], device=torch.device("cpu")
            )
        return KsanaNodeVAEEncodeOutput(
            samples=latent,
            with_end_image=False,
            batch_size_per_prompts=batch_size,
        )

    ksana_engine = get_engine()
    log.info(f"encoder vae: {vae}")
    if isinstance(start_image, torch.Tensor) and start_image.ndim == 3:
        start_image = start_image.unsqueeze(0)
        print(f"start_image{start_image.shape}, {start_image.device}")
    if isinstance(end_image, torch.Tensor) and end_image.ndim == 3:
        end_image = end_image.unsqueeze(0)
        print(f"end_image{end_image.shape}, {end_image.device}")
    channels = 3
    if start_image is not None and start_image.shape[3] == channels:
        start_image = start_image.permute(0, 3, 1, 2)
    if end_image is not None and end_image.shape[3] == channels:
        end_image = end_image.permute(0, 3, 1, 2)

    def preprocess_image(image):
        if image is None:
            return image
        return image.sub(0.5).div(0.5)

    start_image = preprocess_image(start_image)
    end_image = preprocess_image(end_image)

    with_end_image = end_image is not None

    latents = ksana_engine.forward_vae_encode(
        model_key=vae,
        target_f=num_frames,
        target_h=height,
        target_w=width,
        start_img=start_image,
        end_img=end_image,
        mask=mask,
        batch_size=batch_size,
    )
    return KsanaNodeVAEEncodeOutput(
        samples=latents,
        with_end_image=with_end_image,
        batch_size_per_prompts=int(batch_size),
    )


def vae_encode_image(
    vae=None,
    image=None,
    batch_size=None,
):
    if isinstance(image, torch.Tensor):
        image = image.sub(0.5).div(0.5)
    ksana_engine = get_engine()
    log.info(f"encoder vae: {vae}")
    latents = ksana_engine.forward_vae_encode_image(
        model_key=vae,
        image=image,
        batch_size=batch_size,
    )
    return KsanaNodeVAEEncodeOutput(
        samples=latents,
        with_end_image=False,
        batch_size_per_prompts=int(batch_size),
    )


def _comfy_process_output(image):
    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


def vae_decode(vae, latent):
    latents = latent.samples
    with_end_image = latent.with_end_image
    ksana_engine = get_engine()
    log.info(f"latent{latents.shape}, {latents.device}")
    if isinstance(latents, torch.Tensor) and latents.ndim == 4:
        latents = latents.unsqueeze(0)
    images = ksana_engine.forward_vae_decode(
        model_key=vae,
        latents=latents,
        with_end_image=with_end_image,
    )
    images = images.cpu().permute(0, 2, 3, 4, 1)
    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    log.info(f"images {images.shape}, {images.device}")
    return _comfy_process_output(images)
