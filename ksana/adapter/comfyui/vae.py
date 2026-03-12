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
from ksana.models.model_key import get_model_key_from_path
from ksana.nodes.core.node_context import KsanaNodeContext
from ksana.nodes.core.node_types import KsanaInferNodeType
from ksana.tensor import TensorKey
from ksana.utils import get_gpu_count, log
from ksana.utils.profile import MemoryProfiler

from .output_types import KsanaNodeVAEEncodeOutput


class KsanaNodeVAELoader:
    # TODO： 这种方式不安全，如果一个画布有多个相同节点，就会出错，或者反复clear model了
    LOADED_MODEL = None

    @classmethod
    def load(cls, vae_path):
        num_gpus = get_gpu_count()
        ksana_engine = get_engine(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            ksana_engine.clear_models(cls.LOADED_MODEL)
        model_key = get_model_key_from_path(vae_path)
        ksana_engine.run_loader_node(model_key, model_path=vae_path)
        cls.LOADED_MODEL = model_key
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
        # 无 VAE 时也写入 pool，保持 key 模式一致
        ksana_engine = get_engine()
        ksana_engine.put_tensors(**{TensorKey.IMAGE_EMBEDS: [latent]})
        return KsanaNodeVAEEncodeOutput(
            samples=TensorKey.IMAGE_EMBEDS,
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

    context = KsanaNodeContext(
        metadata={
            "target_f": num_frames,
            "target_h": height,
            "target_w": width,
            "mask": mask,
            "batch_size": batch_size,
        }
    )
    with ksana_engine.tensor_scope(keep=[TensorKey.IMAGE_EMBEDS]):
        ksana_engine.put_tensors(**{TensorKey.START_IMG: start_image, TensorKey.END_IMG: end_image})
        ksana_engine.run_infer_node(KsanaInferNodeType.VAE_ENCODE_SPATIAL, vae, context)

    return KsanaNodeVAEEncodeOutput(
        samples=TensorKey.IMAGE_EMBEDS,
        with_end_image=with_end_image,
        batch_size_per_prompts=int(batch_size),
    )


def vae_encode_image(
    vae=None,
    image=None,
    batch_size=None,
):
    MemoryProfiler.record_memory("before vae_encode_image")
    if isinstance(image, torch.Tensor):
        image = image.sub(0.5).div(0.5)
    ksana_engine = get_engine()
    log.info(f"encoder vae: {vae}")

    context = KsanaNodeContext(metadata={"batch_size": batch_size})
    with ksana_engine.tensor_scope(keep=[TensorKey.IMAGE_EMBEDS]):
        ksana_engine.put_tensors(**{TensorKey.IMAGE: image})
        ksana_engine.run_infer_node(KsanaInferNodeType.VAE_ENCODE_IMAGES, vae, context)

    MemoryProfiler.record_memory("after vae_encode_image")
    return KsanaNodeVAEEncodeOutput(
        samples=TensorKey.IMAGE_EMBEDS,
        with_end_image=False,
        batch_size_per_prompts=int(batch_size),
    )


def _comfy_process_output(image):
    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


def vae_decode(vae, latent):
    MemoryProfiler.record_memory("before vae_decode")
    latents_key = latent.samples  # TensorKey — tensor 在 pool 中
    with_end_image = latent.with_end_image
    ksana_engine = get_engine()

    if not ksana_engine.has_tensor(latents_key):
        raise RuntimeError(
            f"vae_decode: tensor key '{latents_key}' not found in pool. "
            "Ensure the upstream node (vae_encode/generate) used tensor_scope(keep=...) correctly."
        )

    context = KsanaNodeContext(metadata={"with_end_image": with_end_image})
    with ksana_engine.tensor_scope():
        # latents_key 可能是 LATENTS 或 IMAGE_EMBEDS，VAEDecodeNode 读 LATENTS
        if latents_key != TensorKey.LATENTS:
            tensor_value = ksana_engine.get_tensor(latents_key)
            ksana_engine.put_tensors(**{TensorKey.LATENTS: tensor_value.data})
        ksana_engine.run_infer_node(KsanaInferNodeType.VAE_DECODE, vae, context)
        video_tv = ksana_engine.get_tensor(TensorKey.VIDEO)
        images = video_tv.data

    images = images.cpu().permute(0, 2, 3, 4, 1)
    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    log.info(f"images {images.shape}, {images.device}")
    MemoryProfiler.record_memory("after vae_decode")
    return _comfy_process_output(images)
