import torch

# 模型大小映射表 (字节)
MODEL_SIZE_MAP = {
    "A14B": 28 * 1024 * 1024 * 1024,  # 14B参数模型，约28GB (fp16)
    "5B": 10 * 1024 * 1024 * 1024,  # 5B参数模型，约10GB (fp16)
}

# 内存使用系数映射表，基于模型名称、任务类型和模型大小
# key格式: (model_name, task_type, model_size)
MEMORY_USAGE_FACTOR_MAP = {
    ("wan2.2", "i2v", "A14B"): 2.305,  # wan_i2v_A14B
    ("wan2.2", "t2v", "A14B"): 2.305,  # wan_t2v_A14B
    ("wan2.2", "s2v", "A14B"): 2.305,  # wan_s2v_14B
    ("wan2.2", "ti2v", "5B"): 1.383,  # wan_ti2v_5B
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
    batch_size = latent_shape[0]

    # latent_shape[0] = prompt 数量，batch_size * 2 表示 cond + uncond 拼接后的批次
    spatial_size = latent_shape[2] * latent_shape[3] * latent_shape[4]

    area_double = (batch_size * 2) * spatial_size
    area_single = batch_size * spatial_size

    dtype_size = torch.tensor([], dtype=run_dtype).element_size()

    coeff = dtype_size * 0.01 * memory_usage_factor * (1024 * 1024)

    memory_required = area_double * coeff
    minimum_memory_required = area_single * coeff

    total_memory_required = model_weight_memory * 1.1 + memory_required
    minimum_memory_needed = model_weight_memory + minimum_memory_required
    return total_memory_required, minimum_memory_needed
