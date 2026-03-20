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

from kdit import get_engine
from kdit.config import RuntimeConfig, SampleConfig, SolverType
from kdit.memory.estimator import (
    MODEL_MEMORY_CONFIG,
    estimate_kdit_model_memory,
    get_available_memory,
)
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_types import InferNodeType
from kdit.tensor import TensorKey
from kdit.utils import log
from kdit.utils.env import KSANA_PROFILE
from kdit.utils.monitor import report
from kdit.utils.profile import MemoryProfiler, TimeProfiler
from kdit.utils.vace import prepare_video_control_config

from .output_types import KsanaNodeGeneratorOutput


def _prepare_memory_for_kdit_models(model_key, latent_shape, run_dtype, comfy_device, comfy_free_mem_func):
    try:
        memory_config = MODEL_MEMORY_CONFIG.get(model_key)
        if memory_config is None:
            raise ValueError(f"Unknown model key: {model_key}")

        model_weight_memory = memory_config["model_size"]
        memory_usage_factor = memory_config["usage_factor"]

        total_memory_required, _ = estimate_kdit_model_memory(
            model_weight_memory, latent_shape, run_dtype, memory_usage_factor
        )
        available_memory = get_available_memory(comfy_device)
        if available_memory < total_memory_required:
            comfy_free_mem_func(total_memory_required, comfy_device, keep_loaded=[])

        log.debug(f"Final free memory: {get_available_memory(comfy_device) / (1024*1024):.1f} MB")
    except Exception as e:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to prepare memory for kDiT models: {e}")


def _resolve_latent_shape_for_memory(kdit_engine, base_latent):
    """从 base_latent 或 aux_latent 推导 latent_shape（仅用于内存预估）。

    Returns:
        latent_shape: list[int] | None
    """

    def _first_tensor_shape(data):
        if isinstance(data, list):
            return list(data[0].shape) if data else None
        return list(data.shape) if data is not None else None

    base_latent_key = base_latent.samples
    if not kdit_engine.has_tensor(base_latent_key):
        raise RuntimeError(
            f"generate: tensor key '{base_latent_key}' not found in pool. "
            "Ensure vae_encode used tensor_scope(keep=...) correctly."
        )
    tensor_value = kdit_engine.get_tensor(base_latent_key)
    raw_data = tensor_value.data if tensor_value is not None else None
    return _first_tensor_shape(raw_data)


@report("comfyui_generate")
def generate(  # noqa: C901
    model,
    positive,
    negative,
    base_latent,
    steps,
    seed,
    aux_latent=None,
    add_noise_to_latent=False,
    scheduler="simple",
    solver_name=SolverType.UNI_PC,
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
        solver_name = SolverType(solver_name)
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
    kdit_engine = get_engine()

    # 启动层级 profiler session（仅在 KSANA_PROFILE=1 时生效）
    _profiler = TimeProfiler.start_session("comfyui_generate") if KSANA_PROFILE else None

    MemoryProfiler.record_memory("before_kdit_engine_generate_with_tensors")

    # 仅用于内存预估
    latent_shape = _resolve_latent_shape_for_memory(kdit_engine, base_latent)

    if comfy_free_mem_func is not None and comfy_device is not None:
        _prepare_memory_for_kdit_models(
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
    batch_size_per_prompts = base_latent.batch_size_per_prompts
    batch_size_per_prompts = [batch_size_per_prompts] * num_prompts

    if sample_shift is not None and float(sample_shift) < 0:
        sample_shift = None

    video_control, control_video_config = prepare_video_control_config(
        video_control_config=video_control_config,
        vace_embeds=vace_embeds,
    )

    # 构建 NodeContext — tensor 参数通过 tensor_pool 传递
    context = NodeContext(
        sample_config=SampleConfig(
            steps=steps,
            cfg_scale=(sample_guide_scale, low_sample_guide_scale),
            shift=sample_shift,
            solver=solver_name,
            denoise=denoise,
            sigmas=sigmas,
            add_noise_to_latent=add_noise_to_latent,
        ),
        runtime_config=RuntimeConfig(
            seed=seed,
            rope_function=rope_function,
            batch_size_per_prompts=batch_size_per_prompts,
        ),
        cache_config=cache_config,
        metadata={
            "video_control": video_control,
            "control_video_config": control_video_config,
            "comfy_bar_callback": comfy_bar_callback,
        },
    )

    with kdit_engine.tensor_scope(keep=[TensorKey.LATENTS]):
        kdit_engine.put_tensors({TensorKey.POSITIVE: positive[0][0], TensorKey.NEGATIVE: negative[0][0]})
        # base_latent.samples 是 BASE_LATENT 或 AUX_LATENT — 已在 pool 中
        if aux_latent is not None and aux_latent.samples is not None:
            # aux_latent.samples 是 TensorKey — 重命名为 GeneratorNode 期望的 AUX_LATENT
            kdit_engine.rename_tensor(aux_latent.samples, TensorKey.AUX_LATENT)
        kdit_engine.run_infer_node(InferNodeType.GENERATE, diffusion_model_key, context)

    MemoryProfiler.record_memory("after_kdit_engine_generate_with_tensors")

    # 结束 profiler session 并打印摘要
    if _profiler is not None:
        _profiler.finish()
        _profiler.print_summary()

    return KsanaNodeGeneratorOutput(
        samples=TensorKey.LATENTS,
        with_end_image=base_latent.with_end_image,
    )
