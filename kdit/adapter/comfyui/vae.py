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

from kdit import get_engine
from kdit.config import DistributedConfig
from kdit.models.model_key import get_model_key_from_path
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.node_types import InferNodeType, IONodeType
from kdit.tensor import TensorKey
from kdit.utils import get_gpu_count, log
from kdit.utils.profile import MemoryProfiler

from .output_types import VAEEncodeOutput


class KsanaNodeVAELoader:
    # TODO： 这种方式不安全，如果一个画布有多个相同节点，就会出错，或者反复clear model了
    LOADED_MODEL = None

    @classmethod
    def load(cls, vae_path):
        num_gpus = get_gpu_count()
        kdit_engine = get_engine(dist_config=DistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            kdit_engine.clear_models(cls.LOADED_MODEL.pin)
        model_key = get_model_key_from_path(vae_path)
        node_def = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=model_key)
        context = NodeContext(metadata={"model_path": vae_path})
        output_pins = kdit_engine.run_node(node_def, {}, context)
        model_pool_key = output_pins.get(model_key)
        cls.LOADED_MODEL = model_pool_key
        return model_pool_key


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
        # TODO: remove me use empty latent
        if width is None or height is None:
            raise ValueError("width/height required if vae is None")
        if num_frames == 1:
            latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=torch.device("cpu"))
        else:
            latent = torch.zeros(
                [batch_size, 16, ((num_frames - 1) // 4) + 1, height // 8, width // 8], device=torch.device("cpu")
            )
        # 无 VAE 时通过 feed_tensors 注入 pool
        kdit_engine = get_engine()
        feed_pins = kdit_engine.feed_tensors({TensorKey.BASE_LATENT: [latent]})
        return VAEEncodeOutput(
            samples=feed_pins[TensorKey.BASE_LATENT],
            with_end_image=False,
            batch_size_per_prompts=batch_size,
        )

    kdit_engine = get_engine()
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

    context = NodeContext(
        metadata={
            "target_f": num_frames,
            "target_h": height,
            "target_w": width,
            "mask": mask,
            "batch_size": batch_size,
        }
    )

    # 注入 tensor 并构建 input_pins
    feed_pins = kdit_engine.feed_tensors({TensorKey.START_IMG: start_image, TensorKey.END_IMG: end_image})
    input_pins = dict(feed_pins)
    input_pins[vae.pin] = vae  # Model pin

    node_def = NodeDef(node_type=InferNodeType.VAE_ENCODE_SPATIAL, model_key=vae.pin)
    result_pins = kdit_engine.run_node(node_def, input_pins, context)

    return VAEEncodeOutput(
        samples=result_pins.get(TensorKey.BASE_LATENT),
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
    kdit_engine = get_engine()
    log.info(f"encoder vae: {vae}")

    context = NodeContext(metadata={"batch_size": batch_size})

    feed_pins = kdit_engine.feed_tensors({TensorKey.IMAGE: image})
    input_pins = dict(feed_pins)
    input_pins[vae.pin] = vae

    node_def = NodeDef(node_type=InferNodeType.VAE_ENCODE_IMAGES, model_key=vae.pin)
    result_pins = kdit_engine.run_node(node_def, input_pins, context)

    MemoryProfiler.record_memory("after vae_encode_image")
    return VAEEncodeOutput(
        samples=result_pins.get(TensorKey.AUX_LATENT),
        with_end_image=False,
        batch_size_per_prompts=int(batch_size),
    )


def _comfy_process_output(image):
    return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


def vae_decode(vae, latent):
    MemoryProfiler.record_memory("before vae_decode")
    latent_pool_key = latent.samples  # TensorPoolKey — tensor 在 pool 中
    with_end_image = latent.with_end_image
    kdit_engine = get_engine()

    if not kdit_engine.has_tensor(latent_pool_key):
        raise RuntimeError(
            f"vae_decode: tensor key '{latent_pool_key}' not found in pool. "
            "Ensure the upstream node (vae_encode/generate) wrote the tensor to the pool correctly."
        )

    context = NodeContext(metadata={"with_end_image": with_end_image})
    try:
        # input_pins: 将 LATENTS 映射到上游的 pool key，无需 rename
        input_pins = {TensorKey.LATENTS: latent_pool_key}
        input_pins[vae.pin] = vae

        node_def = NodeDef(node_type=InferNodeType.VAE_DECODE, model_key=vae.pin)
        result_pins = kdit_engine.run_node(node_def, input_pins, context)

        video_tv = kdit_engine.get_tensor(result_pins[TensorKey.VIDEO])
        images = video_tv.data
    finally:
        kdit_engine.clear_all_tensors()

    images = images.cpu().permute(0, 2, 3, 4, 1)
    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    log.info(f"images {images.shape}, {images.device}")
    MemoryProfiler.record_memory("after vae_decode")
    return _comfy_process_output(images)
