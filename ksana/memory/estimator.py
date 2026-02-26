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
from ksana.models.model_key import KsanaModelKey

# TODO: memory management
# 模型内存配置映射表，基于模型key
MODEL_MEMORY_CONFIG = {
    KsanaModelKey.Wan2_2_T2V_14B: {
        "model_size": 28 * 1024 * 1024 * 1024,  # 14B参数模型，约28GB (fp16)
        "usage_factor": 2.305,  # wan_t2v_A14B
    },
    KsanaModelKey.Wan2_2_I2V_14B: {
        "model_size": 28 * 1024 * 1024 * 1024,  # 14B参数模型，约28GB (fp16)
        "usage_factor": 2.305,  # wan_i2v_A14B
    },
    KsanaModelKey.Wan2_1_VACE_14B: {
        "model_size": 28 * 1024 * 1024 * 1024,  # 14B参数模型，约28GB (fp16)
        "usage_factor": 2.305,  # wan vace 14B
    },
    KsanaModelKey.QwenImage_T2I: {
        "model_size": 40 * 1024 * 1024 * 1024,  # 20B参数模型，约40GB (fp16)
        "usage_factor": 2.5,  # qwen-image t2i 20B
    },
    KsanaModelKey.QwenImage_Edit: {
        "model_size": 40 * 1024 * 1024 * 1024,  # Edit 模式共用 Qwen-Image 20B 模型
        "usage_factor": 2.5,
    },
}

MEMORY_SAFETY_FACTOR = 0.9


def get_available_memory(device: torch.device) -> int:
    if device is None or device.type != "cuda":
        return float("inf")
    dev = device.index if device.index is not None else torch.cuda.current_device()

    stats = torch.cuda.memory_stats(dev)
    mem_active = stats["active_bytes.all.current"]
    mem_reserved = stats["reserved_bytes.all.current"]
    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
    mem_free_torch = mem_reserved - mem_active
    return MEMORY_SAFETY_FACTOR * (mem_free_cuda + mem_free_torch)


def estimate_ksana_model_memory(model_weight_memory, latent_shape, run_dtype, memory_usage_factor=1.0):
    # model_weight_memory 单位是字节
    if len(latent_shape) == 5:
        batch_size = int(latent_shape[0])

        # latent_shape[0] = prompt 数量，batch_size * 2 表示 cond + uncond 拼接后的批次
        spatial_size = latent_shape[2] * latent_shape[3] * latent_shape[4]

        area_double = (batch_size * 2) * spatial_size
        area_single = batch_size * spatial_size

        dtype_size = torch.tensor([], dtype=run_dtype).element_size()
        coeff = dtype_size * 0.01 * memory_usage_factor * (1024 * 1024)

        memory_required = area_double * coeff
        minimum_memory_required = area_single * coeff
    elif len(latent_shape) == 4:
        # 4D latent: [B, C, H, W] does not support COND+UNCOND combined like video latents,
        # so only use area_single
        batch_size = int(latent_shape[0])
        spatial_size = latent_shape[2] * latent_shape[3]
        area_single = batch_size * spatial_size
        dtype_size = torch.tensor([], dtype=run_dtype).element_size()
        coeff = dtype_size * 0.01 * memory_usage_factor * (1024 * 1024)
        memory_required = area_single * coeff
        minimum_memory_required = area_single * coeff
    else:
        raise ValueError(f"Unsupported latent_shape rank: {len(latent_shape)} shape={latent_shape}")

    total_memory_required = model_weight_memory * 1.1 + memory_required
    minimum_memory_needed = model_weight_memory + minimum_memory_required
    return total_memory_required, minimum_memory_needed
