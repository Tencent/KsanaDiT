import os

from ksana import get_generator
from ksana.config import KsanaAttentionConfig, KsanaDistributedConfig, KsanaModelConfig
from ksana.operations import KsanaLinearBackend
from ksana.utils import get_gpu_count, log
from ksana.utils.profile import MemoryProfiler

from .output_types import KsanaComfyModelLoaderOutput


class KsanaComfyModelLoader:
    LOADED_MODEL = None

    @classmethod
    def load(
        cls,
        high_noise_model_path: str,
        low_noise_model_path: str = None,
        run_dtype="float16",
        linear_backend: KsanaLinearBackend | str = KsanaLinearBackend.DEFAULT,
        attention_config: KsanaAttentionConfig | None = None,
        num_gpus="default",
        model_boundary=None,
        torch_compile_args=None,
        lora=None,
        comfy_progress_bar_func=None,
    ):
        num_gpus = get_gpu_count() if num_gpus == "default" else int(num_gpus)
        if comfy_progress_bar_func is None:
            comfyui_progress_bar = None
        else:
            comfyui_progress_bar = comfy_progress_bar_func(1 if low_noise_model_path is None else 2)

        def comfy_bar_callback():
            if comfyui_progress_bar is None:
                return
            comfyui_progress_bar.update(1)

        model_config = KsanaModelConfig(
            run_dtype=run_dtype,
            linear_backend=KsanaLinearBackend(linear_backend),
            attention_config=KsanaAttentionConfig() if attention_config is None else attention_config,
            torch_compile_config=torch_compile_args,
        )

        high_model_loras_list = []
        low_model_loras_list = []
        if lora is not None and isinstance(lora, list) and len(lora) > 0:
            if isinstance(lora[0], list):  # case when lora = [[high_model_loras], [low_model_loras]]
                high_model_loras_list = lora[0]
                if len(lora) > 1:
                    low_model_loras_list = lora[1]
            else:  # case when lora = [high_model_loras]
                high_model_loras_list = lora
        log.info(f"high_model_loras_list: {high_model_loras_list}, low_model_loras_list: {low_model_loras_list}")

        MemoryProfiler.record_memory("before_load_model")

        ksana_generator = get_generator(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            ksana_generator.clear_models(cls.LOADED_MODEL)
        try:
            cls.LOADED_MODEL = ksana_generator.load_diffusion_model(
                model_path=(
                    (high_noise_model_path, low_noise_model_path)
                    if low_noise_model_path is not None
                    else high_noise_model_path
                ),
                model_config=model_config,
                comfy_bar_callback=comfy_bar_callback,
                lora=[high_model_loras_list, low_model_loras_list],
            )
        except Exception as e:
            log.error(f"load_diffusion_model failed, because {e}")
            cls.LOADED_MODEL = None
        MemoryProfiler.record_memory("after_load_model")
        return KsanaComfyModelLoaderOutput(
            model=cls.LOADED_MODEL,
            model_name=os.path.basename(high_noise_model_path),  # TODO(qian): need remove
            run_dtype=model_config.run_dtype,
            boundary=model_boundary,
        )
