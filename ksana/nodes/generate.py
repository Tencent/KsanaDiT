from ksana import get_engine
from ksana.config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverBackend
from ksana.utils import log
from ksana.utils.memory import (
    MODEL_MEMORY_CONFIG,
    estimate_ksana_model_memory,
    get_available_memory,
)
from ksana.utils.profile import MemoryProfiler

from .output_types import KsanaNodeGeneratorOutput


def _prepare_memory_for_ksana_models(model_key, latent_shape, run_dtype, comfy_device, comfy_free_mem_func):
    try:
        memory_config = MODEL_MEMORY_CONFIG.get(model_key)
        if memory_config is None:
            raise ValueError(f"Unknown model key: {model_key}")

        model_weight_memory = memory_config["model_size"]
        memory_usage_factor = memory_config["usage_factor"]

        total_memory_required, _ = estimate_ksana_model_memory(
            model_weight_memory, latent_shape, run_dtype, memory_usage_factor
        )
        available_memory = get_available_memory(comfy_device)
        if available_memory < total_memory_required:
            comfy_free_mem_func(total_memory_required, comfy_device, keep_loaded=[])

        log.debug(f"Final free memory: {get_available_memory(comfy_device) / (1024*1024):.1f} MB")
    except Exception as e:
        raise RuntimeError(f"Failed to prepare memory for KsanaDiT models: {e}")


def generate(
    model,
    positive,
    negative,
    latent_image,
    steps,
    seed,
    scheduler="simple",
    solver_name=KsanaSolverBackend.UNI_PC,
    sample_guide_scale=4.0,
    sample_shift=5.0,
    denoise=1.0,
    rope_function="default",
    low_sample_guide_scale=None,
    cache_configs=None,
    sigmas=None,
    comfy_device=None,
    comfy_progress_bar_func=None,
    comfy_free_mem_func=None,
):
    # Convert string solver_name to enum
    if isinstance(solver_name, str):
        solver_name = KsanaSolverBackend(solver_name)
    if sigmas is not None:
        expected_lengths = steps + 1
        if len(sigmas) != expected_lengths:
            raise RuntimeError(f"sigmas length ({len(sigmas)}) must be equal to steps + 1 ({expected_lengths})")

    ksana_model = model.model
    if ksana_model is None:
        raise RuntimeError(
            "Ksana diffusion model is not loaded (model=None). "
            "Check that `KsanaModelLoaderNode` succeeded and that the requested diffusion model file exists."
        )
    if isinstance(ksana_model, (list, tuple)) and len(ksana_model) == 0:
        raise RuntimeError("Ksana diffusion model key list is empty; model loading likely failed.")
    run_dtype = model.run_dtype
    ksana_engine = get_engine()

    MemoryProfiler.record_memory("before_ksana_engine_generate_with_tensors")
    model_key = ksana_model[0] if isinstance(ksana_model, (list, tuple)) else ksana_model
    latent_shape = latent_image.samples.shape
    if comfy_free_mem_func is not None and comfy_device is not None:
        _prepare_memory_for_ksana_models(
            model_key,
            latent_shape=latent_shape,
            run_dtype=run_dtype,
            comfy_device=comfy_device,
            comfy_free_mem_func=comfy_free_mem_func,
        )
    if comfy_progress_bar_func is not None:
        comfyui_progress_bar = comfy_progress_bar_func(steps)

    def comfy_bar_callback(step, total):
        if comfy_progress_bar_func is not None:
            comfyui_progress_bar.update_absolute(step, total)

    if cache_configs is not None and not isinstance(cache_configs, list):
        cache_configs = [cache_configs]
    num_prompts = positive[0][0].shape[0]
    batch_size_per_prompt = latent_image.batch_size_per_prompt
    batch_size_per_prompt = [batch_size_per_prompt] * num_prompts

    if sample_shift is not None and float(sample_shift) < 0:
        sample_shift = None
    # TODO: maybe need to latent_format process_in for positive/negative?
    samples = ksana_engine.forward_diffusion_models_with_tensors(
        model_keys=ksana_model,
        positive=positive[0][0],
        negative=negative[0][0],
        img_latents=latent_image.samples,  # [1, 16, 5, h/, w/]
        sample_config=KsanaSampleConfig(
            steps=steps,
            cfg_scale=(sample_guide_scale, low_sample_guide_scale),
            shift=sample_shift,
            solver=solver_name,
            denoise=denoise,
            sigmas=sigmas,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=seed,
            rope_function=rope_function,
            batch_size_per_prompt=batch_size_per_prompt,
        ),
        cache_configs=cache_configs,
        comfy_bar_callback=comfy_bar_callback,
    )
    MemoryProfiler.record_memory("after_ksana_engine_generate_with_tensors")

    # Note: only rank 0 return tensor, so maybe None
    if samples is not None and len(samples.shape) == 4:
        samples = samples.unsqueeze(0)

    return KsanaNodeGeneratorOutput(
        samples=samples,
        with_end_image=latent_image.with_end_image,
    )
