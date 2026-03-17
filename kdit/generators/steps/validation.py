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

"""Generator 验证函数 — 从 BaseGenerator 提取的无状态校验逻辑。"""

import torch

from kdit.config import KsanaSampleConfig, KsanaSolverType, RuntimeConfig
from kdit.config.cache_config import HybridCacheConfig, KsanaCacheConfig, warp_as_hybrid_cache
from kdit.models import KsanaDiffusionModel, ModelKey
from kdit.utils import evolve_with_recommend, log


def valid_diffusion_model(
    diffusion_model: KsanaDiffusionModel | list[KsanaDiffusionModel],
    model_key: ModelKey,
) -> list[KsanaDiffusionModel]:
    """校验并规范化 diffusion_model 为 list 形式。"""
    if isinstance(diffusion_model, (tuple, list)):
        diffusion_model = list(diffusion_model)
    elif isinstance(diffusion_model, KsanaDiffusionModel):
        diffusion_model = [diffusion_model]
    else:
        raise ValueError(
            f"diffusion_model {diffusion_model} must be KsanaDiffusionModel or list of KsanaDiffusionModel"
        )
    if len(diffusion_model) != 1:
        if model_key in [ModelKey.Wan2_2_I2V_14B, ModelKey.Wan2_2_T2V_14B]:
            if len(diffusion_model) > 2 or len(diffusion_model) < 1:
                raise ValueError(f"{model_key} must have one or two model, but got {len(diffusion_model)} model")
            else:
                if model_key != diffusion_model[0].model_key or model_key != diffusion_model[1].model_key:
                    raise ValueError(f"{model_key} must match but got {diffusion_model[0].model_key}")
                if diffusion_model[0].run_dtype != diffusion_model[1].run_dtype:
                    raise ValueError(
                        f"{model_key} must have same run_dtype, but got "
                        f"{diffusion_model[0].run_dtype} and {diffusion_model[1].run_dtype}"
                    )
        else:
            raise ValueError(f"{model_key} must have only one model, but got {len(diffusion_model)} model")
    return diffusion_model


def valid_sample_config(sample_config: KsanaSampleConfig, model_len: int) -> KsanaSampleConfig:
    """校验并规范化 sample_config。"""
    log.info(f"sample_config: {sample_config}")
    if isinstance(sample_config.cfg_scale, (float, int)):
        sample_config = evolve_with_recommend(
            sample_config, {"cfg_scale": [sample_config.cfg_scale] * model_len}, force_update=True
        )
    elif isinstance(sample_config.cfg_scale, (list, tuple)):
        if len(sample_config.cfg_scale) < model_len:
            raise ValueError(f"cfg_scale length must be {model_len}, but got {len(sample_config.cfg_scale)}")
        sample_config = evolve_with_recommend(
            sample_config, {"cfg_scale": list(sample_config.cfg_scale)}, force_update=True
        )
    else:
        raise TypeError(f"sample_config.cfg_scale {sample_config.cfg_scale} type not supported")
    if sample_config.solver is None or not KsanaSolverType.support(sample_config.solver):
        raise ValueError(f"sample_config.solver must in support list {KsanaSolverType.get_supported_list()}")
    if sample_config.denoise <= 0.0:
        raise ValueError(f"denoise <= 0.0 is not supported, got {sample_config.denoise}")
    return sample_config


def valid_cache_config(cache_config: KsanaCacheConfig | HybridCacheConfig, model_len: int) -> HybridCacheConfig:
    """校验并规范化 cache_config 为 HybridCacheConfig list。"""
    log.info(f"cache_config: {cache_config}")
    if cache_config is None:
        return
    if not (len(cache_config) == 1 or len(cache_config) == model_len):
        raise ValueError(f"cache_config length must be {model_len} or 1, but got {len(cache_config)}")
    hybrid_caches = []
    for i in range(model_len):
        cache_id = min(i, len(cache_config) - 1)  # allow two model use same cache config
        one_config = cache_config[cache_id]
        if one_config is None:
            hybrid_caches.append(None)
            continue
        if not isinstance(one_config, (KsanaCacheConfig, HybridCacheConfig)):
            raise ValueError(f"cache_config {one_config} must be KsanaCacheConfig or HybridCacheConfig")
        as_hybrid_cache = warp_as_hybrid_cache(one_config)
        hybrid_caches.append(as_hybrid_cache)
    return hybrid_caches


def valid_runtime_config(runtime_config: RuntimeConfig, num_prompts: int) -> RuntimeConfig:
    """校验并规范化 runtime_config。"""
    log.info(f"runtime_config: {runtime_config}")
    if runtime_config is None:
        raise ValueError("runtime_config must be provided")
    batch_size_per_prompts = runtime_config.batch_size_per_prompts
    if batch_size_per_prompts is None:
        batch_size_per_prompts = [1] * num_prompts
    elif isinstance(batch_size_per_prompts, int):
        batch_size_per_prompts = [batch_size_per_prompts] * num_prompts
    elif isinstance(batch_size_per_prompts, (list, tuple)):
        if len(batch_size_per_prompts) != num_prompts:
            raise ValueError(
                f"batch_size_per_prompts({batch_size_per_prompts}) len " f"must match num_prompts ({num_prompts})"
            )
    else:
        raise TypeError(f"batch_size_per_prompts must be int/list[int]/None, but got {type(batch_size_per_prompts)}")
    runtime_config = evolve_with_recommend(
        runtime_config,
        {"batch_size_per_prompts": batch_size_per_prompts},
        force_update=True,
    )
    return runtime_config


def valid_input_latent(input_latent: torch.Tensor, noise_shape: tuple[int]):
    """校验 input_latent 与 noise_shape 的维度一致性。"""
    if input_latent is None:
        return
    if input_latent.dim() != len(noise_shape) or len(noise_shape) != 5:  # [bs, z_dim, f, h, w]
        raise ValueError(
            f"input_latent.dim() {input_latent.dim()} must be equal to noise_shape.len()"
            f" {len(noise_shape)} and both must be 5"
        )
    input_bs, input_z_dim, _, input_h, input_w = input_latent.shape
    noise_bs, noise_z_dim, _, noise_h, noise_w = noise_shape
    if input_bs != noise_bs or input_z_dim != noise_z_dim or input_h != noise_h or input_w != noise_w:
        raise ValueError(
            f"input_latent shape {input_latent.shape} must match "
            f" noise_shape {noise_shape} in all dimensions except frame dimension"
        )
