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

from ksana import get_engine
from ksana.config import KsanaRuntimeConfig, KsanaSampleConfig, KsanaSolverType
from ksana.memory.estimator import (
    MODEL_MEMORY_CONFIG,
    estimate_ksana_model_memory,
    get_available_memory,
)
from ksana.models.model_key import KsanaModelKey
from ksana.utils import log
from ksana.utils.monitor import report
from ksana.utils.profile import MemoryProfiler
from ksana.utils.vace import prepare_video_control_config

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
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to prepare memory for KsanaDiT models: {e}")


@report("comfyui_generate")
def generate(
    model,
    positive,
    negative,
    image_embeds,
    steps,
    seed,
    latent=None,
    add_noise_to_latent=False,
    scheduler="simple",
    solver_name=KsanaSolverType.UNI_PC,
    sample_guide_scale=4.0,
    sample_shift=5.0,
    denoise=1.0,
    rope_function="default",
    low_sample_guide_scale=None,
    cache_config=None,
    sigmas=None,
    video_control_config=None,
    vace_embeds=None,
    comfy_device=None,
    comfy_progress_bar_func=None,
    comfy_free_mem_func=None,
):
    # Convert string solver_name to enum
    if isinstance(solver_name, str):
        solver_name = KsanaSolverType(solver_name)
    if sigmas is not None:
        expected_lengths = steps + 1
        if len(sigmas) != expected_lengths:
            raise RuntimeError(f"sigmas length ({len(sigmas)}) must be equal to steps + 1 ({expected_lengths})")

    diffusion_model_key = model.model
    if diffusion_model_key is None:
        raise RuntimeError(
            "Ksana diffusion model is not loaded (model=None). "
            "Check that `KsanaModelLoaderNode` succeeded and that the requested diffusion model file exists."
        )
    if isinstance(diffusion_model_key, (list, tuple)):
        raise RuntimeError("Ksana diffusion model key can not be list or tuple.")
    run_dtype = model.run_dtype
    ksana_engine = get_engine()

    MemoryProfiler.record_memory("before_ksana_engine_generate_with_tensors")

    # For memory estimation, use latent shape if available, otherwise infer from image_embeds
    if latent is not None:
        latent_shape = latent.samples.shape
    elif isinstance(image_embeds.samples, list):
        latent_shape = image_embeds.samples[0].shape
    else:
        latent_shape = image_embeds.samples.shape
    if comfy_free_mem_func is not None and comfy_device is not None:
        _prepare_memory_for_ksana_models(
            diffusion_model_key,
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

    if cache_config is not None and not isinstance(cache_config, list):
        cache_config = [cache_config]
    num_prompts = positive[0][0].shape[0]
    batch_size_per_prompts = image_embeds.batch_size_per_prompts
    batch_size_per_prompts = [batch_size_per_prompts] * num_prompts

    if sample_shift is not None and float(sample_shift) < 0:
        sample_shift = None

    video_control, control_video_config = prepare_video_control_config(
        video_control_config=video_control_config,
        vace_embeds=vace_embeds,
    )
    noise_shape = None
    img_latents = image_embeds.samples  # [bs, 16, 5, h/, w/] or list[Tensor]
    # Qwen Image Edit: latent 直接决定输出 shape，不需要修正
    # Wan I2V (wan2.1_vace_add_noise_i2v_test.json): latent 来自 KsanaVAEImageEncodeNode（单帧），
    # TODO(qiannan): 需要进一步明确各 workflow 中 latent 输入的语义
    if latent is not None and diffusion_model_key == KsanaModelKey.QwenImage_Edit:
        noise_shape = list(latent.samples.shape[1:])
    elif diffusion_model_key == KsanaModelKey.QwenImage_T2I:
        # T2I: image_embeds 仅用于提供输出 shape，不作为图像条件传入 generator
        first_sample = img_latents[0] if isinstance(img_latents, list) else img_latents
        noise_shape = list(first_sample.shape[1:])
        img_latents = None

    # TODO: maybe need to latent_format process_in for positive/negative?
    samples = ksana_engine.forward_generator(
        model_key=diffusion_model_key,
        noise_shape=noise_shape,
        positive=positive[0][0],
        negative=negative[0][0],
        img_latents=img_latents,
        input_latent=latent.samples if latent is not None else None,
        sample_config=KsanaSampleConfig(
            steps=steps,
            cfg_scale=(sample_guide_scale, low_sample_guide_scale),
            shift=sample_shift,
            solver=solver_name,
            denoise=denoise,
            sigmas=sigmas,
            add_noise_to_latent=add_noise_to_latent,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=seed,
            rope_function=rope_function,
            batch_size_per_prompts=batch_size_per_prompts,
        ),
        cache_config=cache_config,
        comfy_bar_callback=comfy_bar_callback,
        video_control=video_control,
        control_video_config=control_video_config,
    )
    MemoryProfiler.record_memory("after_ksana_engine_generate_with_tensors")

    # Note: only rank 0 return tensor, so maybe None
    if samples is not None and len(samples.shape) == 4:
        samples = samples.unsqueeze(0)

    return KsanaNodeGeneratorOutput(
        samples=samples,
        with_end_image=image_embeds.with_end_image,
    )
