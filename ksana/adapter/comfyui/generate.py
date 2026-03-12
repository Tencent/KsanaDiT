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
from ksana.nodes.core.node_context import KsanaNodeContext
from ksana.nodes.core.node_types import KsanaInferNodeType
from ksana.tensor import TensorKey
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


def _resolve_latent_shape(ksana_engine, image_embeds, latent, diffusion_model_key):
    """从 image_embeds key 或 latent 推导 latent_shape 和 image_embeds_list。

    Returns:
        (latent_shape, image_embeds_list)
        - latent_shape: list[int] | None
        - image_embeds_list: list[Tensor] | None — 裸 tensor list，用于 put_tensors
    """
    noise_shape = None
    image_embeds_list = None

    # image_embeds.samples 是 TensorKey — 从 pool 取裸 tensor
    image_embeds_key = image_embeds.samples
    if not ksana_engine.has_tensor(image_embeds_key):
        raise RuntimeError(
            f"generate: tensor key '{image_embeds_key}' not found in pool. "
            "Ensure vae_encode used tensor_scope(keep=...) correctly."
        )
    tensor_value = ksana_engine.get_tensor(image_embeds_key)
    raw_data = tensor_value.data if tensor_value is not None else None

    if isinstance(raw_data, list):
        image_embeds_list = raw_data
    elif raw_data is not None:
        image_embeds_list = [raw_data]

    # latent shape 推导
    if latent is not None:
        latent_key = latent.samples  # TensorKey
        tensor_value = ksana_engine.get_tensor(latent_key)
        latent_raw = tensor_value.data if tensor_value is not None else None
        latent_shape = list(latent_raw.shape) if latent_raw is not None else None
    elif image_embeds_list is not None and len(image_embeds_list) > 0:
        latent_shape = list(image_embeds_list[0].shape)
    else:
        latent_shape = None

    # Qwen Image Edit: latent 直接决定输出 shape
    if latent is not None and diffusion_model_key == KsanaModelKey.QwenImage_Edit:
        tensor_value = ksana_engine.get_tensor(latent.samples)
        latent_raw = tensor_value.data if tensor_value is not None else None
        noise_shape = list(latent_raw.shape[1:]) if latent_raw is not None else None
    elif diffusion_model_key == KsanaModelKey.QwenImage_T2I:
        # T2I: image_embeds 仅用于提供输出 shape，不作为图像条件传入 generator
        if image_embeds_list is not None and len(image_embeds_list) > 0:
            noise_shape = list(image_embeds_list[0].shape[1:])
        image_embeds_list = None

    return noise_shape, image_embeds_list, latent_shape


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

    # 从 pool 中的 key 推导 shape 和裸 tensor
    noise_shape, image_embeds_list, latent_shape = _resolve_latent_shape(
        ksana_engine, image_embeds, latent, diffusion_model_key
    )

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

    # 构建 KsanaNodeContext — tensor 参数通过 tensor_pool 传递
    context = KsanaNodeContext(
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
        metadata={
            "noise_shape": noise_shape,
            "video_control": video_control,
            "control_video_config": control_video_config,
            "comfy_bar_callback": comfy_bar_callback,
        },
    )

    with ksana_engine.tensor_scope(keep=[TensorKey.LATENTS]):
        ksana_engine.put_tensors(**{TensorKey.POSITIVE: positive[0][0], TensorKey.NEGATIVE: negative[0][0]})
        if image_embeds_list is not None:
            ksana_engine.put_tensors(**{TensorKey.IMAGE_EMBEDS: image_embeds_list})
        if latent is not None and latent.samples is not None:
            # latent.samples 是 TensorKey — 重命名为 GeneratorNode 期望的 INPUT_LATENT
            ksana_engine.rename_tensor(latent.samples, TensorKey.INPUT_LATENT)
        ksana_engine.run_infer_node(KsanaInferNodeType.GENERATE, diffusion_model_key, context)

    MemoryProfiler.record_memory("after_ksana_engine_generate_with_tensors")

    return KsanaNodeGeneratorOutput(
        samples=TensorKey.LATENTS,
        with_end_image=image_embeds.with_end_image,
    )
