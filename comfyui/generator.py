import comfy
import comfy.model_management as mm
from ksana import get_generator
from ksana.config import KsanaSampleConfig, KsanaRuntimeConfig
from ksana.utils.profile import MemoryProfiler
from ksana.utils import log


class KsanaGeneratorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "KSANAMODEL",
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "The conditioning describing the attributes you want to include in the image."},
                ),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "The conditioning describing the attributes you want to exclude from the image."},
                ),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "steps": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        # "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "solver_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {
                        "default": "uni_pc",
                        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
                    },
                ),
                "sample_guide_scale": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "sample_shift": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                # "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
                    },
                ),
            },
            "optional": {
                "low_model": (
                    "KSANAMODEL",
                    {"tooltip": "The model used for denoising the input latent."},
                ),
                "boundary": (
                    "FLOAT",
                    {
                        "default": 0.875,
                        "min": 0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.01,
                        "tooltip": "The boundary value used for high and low timesteps.",
                    },
                ),
                "low_sample_guide_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
                "high_cache_config": (
                    "KSANA_CACHE_CONFIG",
                    {"tooltip": "The cache used for high model."},
                ),
                "low_cache_config": (
                    "KSANA_CACHE_CONFIG",
                    {"tooltip": "The cache used for low model."},
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    CATEGORY = "ksana"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def _estimate_ksana_model_memory(self, ksana_model, latent_shape):
        """
        估算KsanaDiT模型需要的内存
        参考 comfy/sampler_helpers.py 的 estimate_memory
        """
        # 1. 计算模型权重大小（类似 model.model_size()）
        model_weight_memory = 0
        for param in ksana_model.model.parameters():
            model_weight_memory += param.nelement() * param.element_size()

        # 2. 获取 comfyui_memory_usage_factor (从 ksana_model.default_model_config 中获取)
        model_config = ksana_model.default_model_config
        if model_config is not None and hasattr(model_config, "comfyui_memory_usage_factor"):
            memory_usage_factor = model_config.comfyui_memory_usage_factor
        else:
            memory_usage_factor = 1.0
            log.warning(
                f"[_estimate_ksana_model_memory] comfyui_memory_usage_factor not found in model config, using default 1.0. "
                f"Model: {type(ksana_model).__name__}"
            )
        # 3. 计算推理过程中的内存需求（激活值等）
        # 参考 model_base.py 的 memory_required 方法
        # 原始公式: area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))
        # 其中 input_shape[0] 是 batch, input_shape[2:] 是空间维度 [T, H, W]
        # 重要：batch_size 要乘以 2，因为推理时同时处理 cond 和 uncond
        batch_doubled = latent_shape[0] * 2  # cond + uncond
        spatial_size = latent_shape[2] * latent_shape[3] * latent_shape[4]  # T * H * W
        area = batch_doubled * spatial_size

        # 注意: 使用 model.run_dtype 而不是 model.dtype
        # model.dtype 可能返回 float32 (因为 patch_embedding 是 float32)，但实际推理使用的是 model.run_dtype (在 model_loader 中设置的)
        actual_dtype = (
            ksana_model.run_dtype
            if hasattr(ksana_model, "run_dtype") and ksana_model.run_dtype is not None
            else ksana_model.dtype
        )
        dtype_size = mm.dtype_size(actual_dtype)

        # 标准注意力机制下的内存需求：memory_required = (area * 0.15 * memory_usage_factor) * (1024 * 1024)
        # 走xformer 或者 flash attention（默认走flash attention）
        memory_required = (area * dtype_size * 0.01 * memory_usage_factor) * (1024 * 1024)

        # 最小内存需求：使用 batch_size = latent_shape[0]（不乘2）
        # 对应 estimate_memory 中的 [noise_shape[0]] + list(noise_shape[1:])
        batch_normal = latent_shape[0]
        min_area = batch_normal * spatial_size

        # 走xformer 或者 flash attention（默认走flash attention）
        minimum_memory_required = (min_area * dtype_size * 0.01 * memory_usage_factor) * (1024 * 1024)

        log.debug("[_estimate_ksana_model_memory]")
        log.debug(f"  latent_shape: {latent_shape}")
        log.debug(f"  memory_usage_factor: {memory_usage_factor}")
        log.debug(f"  area (batch*2): {area}, min_area (batch*1): {min_area}")
        log.debug(f"  actual_dtype (for calculation): {actual_dtype}, dtype_size: {dtype_size} bytes")
        log.debug(f"  model.dtype: {ksana_model.dtype}, model.run_dtype: {getattr(ksana_model, 'run_dtype', 'N/A')}")
        log.debug(f"  memory_required: {memory_required} ({memory_required/(1024**3):.2f} GB)")
        log.debug(f"  minimum_memory_required: {minimum_memory_required} ({minimum_memory_required/(1024**3):.2f} GB)")
        log.debug(f"  model_weight_memory: {model_weight_memory} ({model_weight_memory/(1024**3):.2f} GB)")
        return memory_required, minimum_memory_required, model_weight_memory

    def _prepare_memory_for_ksana_models(self, high_model, low_model, latent_shape, device):
        """
        在加载KsanaDiT模型前，释放ComfyUI模型占用的内存
        类似于 comfy/sampler_helpers.py 的 _prepare_sampling 逻辑
        核心思想：根据KsanaDiT模型实际需要的内存量，智能释放ComfyUI模型
        """
        try:
            # 计算两个模型总共需要的内存（因为在推理过程中两个模型都会加载）
            models_to_check = [high_model]
            if low_model is not None:
                models_to_check.append(low_model)

            max_model_weight_memory = 0
            max_memory_required = 0
            max_minimum_memory_required = 0

            for ksana_model in models_to_check:
                memory_required, minimum_memory_required, model_weight_memory = self._estimate_ksana_model_memory(
                    ksana_model, latent_shape
                )
                # 使用最大的推理内存需求（因为是顺序推理，不是同时推理）
                max_model_weight_memory = max(max_model_weight_memory, model_weight_memory)
                max_memory_required = max(max_memory_required, memory_required)
                max_minimum_memory_required = max(max_minimum_memory_required, minimum_memory_required)

            # 获取当前可用内存
            current_free_mem = mm.get_free_memory(device)
            # 获取extra_reserved_memory（类似_prepare_sampling中的extra_mem）
            extra_mem = mm.extra_reserved_memory()
            inference_memory = mm.minimum_inference_memory()
            extra_mem = max(inference_memory, extra_mem)

            # 计算需要释放的内存量
            # 参考 load_models_gpu 的逻辑：
            # 1. total_memory_required = model_weight_memory (模型权重需要的内存)
            # 2. 实际需要的内存 = model_weight_memory * 1.1 + memory_required + extra_mem
            # 3. 最小需求 = model_weight_memory + minimum_memory_required + extra_mem
            total_memory_required = max_model_weight_memory * 1.1 + max_memory_required + extra_mem
            minimum_memory_needed = max_model_weight_memory + max_minimum_memory_required + extra_mem

            log.debug("KsanaDiT models memory estimate:")
            log.debug(f"  latent_shape={latent_shape}")
            log.debug(f"  extra_mem: {extra_mem / (1024*1024*1024):.1f} GB")
            log.debug(f"  inference_memory: {inference_memory / (1024*1024*1024):.1f} GB")
            log.debug(f"  max_model_weight_memory: {max_model_weight_memory / (1024*1024*1024):.1f} GB")
            log.debug(f"  max_memory_required: {max_memory_required / (1024*1024*1024):.1f} GB")
            log.debug(f"  max_minimum_memory_required: {max_minimum_memory_required / (1024*1024*1024):.1f} GB")
            log.debug(f"  total_memory_required: {total_memory_required / (1024*1024*1024):.1f} GB")
            log.debug(f"  minimum_memory_needed: {minimum_memory_needed / (1024*1024*1024):.1f} GB")
            log.debug(f"  current_free_mem: {current_free_mem / (1024*1024*1024):.1f} GB")

            # 如果当前可用内存不足，释放ComfyUI模型
            # 参考 load_models_gpu 中的两步释放策略
            if current_free_mem < total_memory_required:
                log.debug(f"Need to free {(total_memory_required - current_free_mem) / (1024*1024):.1f} MB")
                mm.free_memory(total_memory_required, device, keep_loaded=[])

            # 如果可用内存仍然小于最小需求，再次尝试释放
            free_mem = mm.get_free_memory(device)
            if free_mem < minimum_memory_needed:
                log.debug("Still need more memory, trying to free minimum required")
                models_unloaded = mm.free_memory(minimum_memory_needed, device, keep_loaded=[])
                log.debug(f"{len(models_unloaded)} ComfyUI models unloaded.")

            # 打印最终内存状态
            final_free_mem = mm.get_free_memory(device)
            log.debug(f"Memory freed: {(final_free_mem - current_free_mem) / (1024*1024):.1f} MB")
            log.debug(f"Final free memory: {final_free_mem / (1024*1024):.1f} MB")
        except Exception as e:
            # 如果释放失败，记录错误但不影响后续流程
            log.warning(f"Failed to prepare memory for KsanaDiT models: {e}")

    def run(
        self,
        model,
        positive,
        negative,
        latent_image,
        steps,
        seed,
        solver_name,
        sample_guide_scale,
        sample_shift,
        denoise=1.0,
        low_model=None,
        boundary=None,
        low_sample_guide_scale=None,
        high_cache_config=None,
        low_cache_config=None,
    ):
        # model.model.keys()?
        # unet = UNet2DConditionModel.from_config(config)
        # unet.load_state_dict(my_state_dict)?
        ksana_generator = get_generator()
        MemoryProfiler.record_memory("before_ksana_generator_generate_video_with_tensors")

        # 在调用 generate_video_with_tensors 前释放 ComfyUI 模型占用的内存
        latent_shape = latent_image["samples"].shape
        device = mm.get_torch_device()
        self._prepare_memory_for_ksana_models(
            high_model=model.model.ksana_model,
            low_model=low_model.model.ksana_model if low_model is not None else None,
            latent_shape=latent_shape,
            device=device,
        )

        # TODO: maybe need to latent_format process_in for positive/negative?
        samples = ksana_generator.generate_video_with_tensors(
            high_model=model.model.ksana_model,
            positive=positive[0][0],  # 1, 512, 4096?
            negative=negative[0][0],  # 1, 512, 4096?
            latents=latent_image["samples"],  # [1, 16, 5, h/, w/]
            sample_config=KsanaSampleConfig(
                steps=steps,
                cfg_scale=(sample_guide_scale, low_sample_guide_scale),
                shift=sample_shift,
                solver=solver_name,
                denoise=denoise,
            ),
            runtime_config=KsanaRuntimeConfig(
                seed=seed,
                boundary=boundary,
            ),
            low_model=low_model.model.ksana_model if low_model is not None else None,
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
        )
        MemoryProfiler.record_memory("after_ksana_generator_generate_video_with_tensors")
        if len(samples.shape) == 4:
            samples = samples.unsqueeze(0)
        samples = model.model.model_config.latent_format.process_out(samples)
        # out = latent_image.copy()
        return ({"samples": samples},)
