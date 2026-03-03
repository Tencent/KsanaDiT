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

from abc import ABC
from functools import partial

import torch

from ..accelerator import platform
from ..config import KsanaDistributedConfig
from ..distributed import shard_model
from ..models.model_key import KsanaModelKey, get_model_key_from_path
from ..models.model_pool import KsanaModelPool
from ..units import KsanaUnitFactory, KsanaUnitType
from ..utils import log, time_range
from ..utils.logger import reset_logging

if platform.is_npu():
    import torch_npu  # pylint: disable=unused-import # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # pylint: disable=unused-import # noqa: F401


class KsanaExecutor(ABC):
    """
    Base class for all Ksana executors.
    和模型有关的配置信息不放在Executor中，而是放在KsanaModel中
    这里只放和device，分布式相关的信息
    """

    def __init__(self, device_id: int = 0, offload_device: str = "cpu"):
        """
        Initialize the executor.
        """
        self.device_id = device_id
        self.rank_id = device_id
        self.world_size = 1
        self.device = torch.device(f"cuda:{self.device_id}")
        self.offload_device = torch.device(offload_device)
        torch.cuda.set_device(self.device)
        # Note: each executor has its own model pool for nodes call, and pipeline own engine then can use executors
        self.model_pool = KsanaModelPool()
        self.local_pipeline = None
        self.shard_fn = None
        self.dist_config = KsanaDistributedConfig(num_gpus=1, use_sp=False, dit_fsdp=False, ulysses_size=1)
        log.info(f"create executor with device_id {self.device_id}, offload_device {self.offload_device}")
        reset_logging()

    def init_torch_dist_group(self, rank_id, dist_config: KsanaDistributedConfig):
        """r initialize sequence parallel group."""
        self.dist_config = dist_config
        log.info(f"init torch dist group with dist_config {dist_config}")
        if dist_config.num_gpus <= 1:
            return
        self.rank_id = rank_id
        self.world_size = dist_config.num_gpus
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if platform.is_gpu() else "hccl",
                init_method="env://",
                rank=rank_id,
                device_id=self.device,
                world_size=dist_config.num_gpus,
            )
        log.info(f"init distributed group with rank_id {self.rank_id}, world_size {self.world_size}")
        reset_logging(rank_id)
        self.shard_fn = partial(shard_model, device_id=self.device_id) if self.dist_config.dit_fsdp else None

    def load_diffusion_model(
        self,
        model_path,
        model_key: KsanaModelKey = None,
        **kwargs,
    ) -> KsanaModelKey:
        model_key = get_model_key_from_path(model_path) if model_key is None else model_key
        model_loader = KsanaUnitFactory.create(KsanaUnitType.LOADER, model_key)
        loaded_model = model_loader.run(
            model_path=model_path,
            dist_config=self.dist_config,
            device=self.offload_device,  # always load to cpu at first
            offload_device=self.offload_device,
            shard_fn=self.shard_fn,
            **kwargs,
        )
        key_matched = True
        if isinstance(loaded_model, (list, tuple)):
            key_matched = all(model_key == model.model_key for model in loaded_model)
        else:
            key_matched = model_key == loaded_model.model_key
        if not key_matched:
            key = loaded_model[0].model_key if isinstance(loaded_model, (list, tuple)) else loaded_model.model_key
            raise RuntimeError(f"model_key {model_key} not match with model {key}")
        self.model_pool.update_model_with_key(model_key, loaded_model)
        return model_key

    def load_text_encoder(self, model_path, model_key: KsanaModelKey = None) -> KsanaModelKey:
        model_key = get_model_key_from_path(model_path) if model_key is None else model_key
        model_loader = KsanaUnitFactory.create(KsanaUnitType.LOADER, model_key)
        loaded_text_encoder = model_loader.run(
            model_path=model_path, device=self.offload_device, shard_fn=self.shard_fn
        )
        self.model_pool.update_model(loaded_text_encoder)
        if loaded_text_encoder.model_key != model_key:
            raise RuntimeError(f"model_key {model_key} should match with model {loaded_text_encoder.model_key}")
        return loaded_text_encoder.model_key

    def load_vae_model(self, model_path, model_key: KsanaModelKey = None) -> KsanaModelKey:
        model_key = get_model_key_from_path(model_path) if model_key is None else model_key
        model_loader = KsanaUnitFactory.create(KsanaUnitType.LOADER, model_key)
        loaded_vae = model_loader.run(model_path=model_path, device=self.offload_device, shard_fn=self.shard_fn)
        self.model_pool.update_model(loaded_vae)
        if loaded_vae.model_key != model_key:
            raise RuntimeError(f"model_key {model_key} should match with model {loaded_vae.model_key}")
        return loaded_vae.model_key

    def forward_text_encode(self, text_encoder_key, **kwargs):
        text_encoder_model = self.model_pool.get_model(text_encoder_key)
        text_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, text_encoder_key)
        return text_encoder.run(text_encoder_model, device=self.device, **kwargs)

    def forward_vae_encode(self, model_key, **kwargs):
        vae_model = self.model_pool.get_model(model_key)
        vae_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, model_key)
        return vae_encoder.run(vae_model, device=self.device, **kwargs)

    @time_range
    def forward_vae_encode_image(self, model_key, **kwargs):
        vae_model = self.model_pool.get_model(model_key)
        vae_encoder = KsanaUnitFactory.create(KsanaUnitType.ENCODER, model_key)
        return vae_encoder.run_encode_image(vae_model, device=self.device, **kwargs)

    @time_range
    def forward_vae_decode(self, model_key, **kwargs):
        vae_model = self.model_pool.get_model(model_key)
        vae_decoder = KsanaUnitFactory.create(KsanaUnitType.DECODER, model_key)
        return vae_decoder.run(vae_model, local_rank=self.rank_id, device=self.device, **kwargs)

    @time_range
    def forward_generator(
        self,
        model_key,
        *,
        noise_shape: list[int] = None,
        img_latents: torch.Tensor | list[torch.Tensor] = None,
        **kwargs,
    ):
        diffusion_model = self.model_pool.get_model(model_key)
        if noise_shape is None and img_latents is None:
            raise ValueError("noise_shape or img_latents must be provided at least one")
        if noise_shape is None:
            # 支持 list[Tensor]（Edit 模式多 prompt）
            first_latent = img_latents[0] if isinstance(img_latents, list) else img_latents
            noise_shape = list(first_latent.shape[1:])
        generator = KsanaUnitFactory.create(KsanaUnitType.GENERATOR, model_key)
        latents = generator.run(
            diffusion_model=diffusion_model,
            noise_shape=noise_shape,
            img_latents=img_latents,
            device=self.device,
            offload_device=self.offload_device,
            **kwargs,
        )
        # only resturn latents on rank 0, since all rank have the same latents
        return latents if self.rank_id == 0 else None

    def clear_models(self, model_keys: list[KsanaModelKey] | KsanaModelKey = None):
        if self.model_pool is None:
            return
        self.model_pool.clear_models(model_keys)

    def generate(self, *args, **kwargs):
        if self.local_pipeline is None:
            raise RuntimeError("local_pipeline is not loaded, call load_models firstly")
        self.local_pipeline.generate(
            *args, local_rank=self.rank_id, device=self.device, offload_device=self.offload_device, **kwargs
        )
