import os

from ksana import get_engine
from ksana.config import KsanaAttentionConfig, KsanaDistributedConfig, KsanaModelConfig
from ksana.operations import KsanaLinearBackend
from ksana.utils import get_gpu_count, log
from ksana.utils.profile import MemoryProfiler

from .output_types import KsanaNodeModelLoaderOutput


class KsanaNodeModelLoader:
    LOADED_MODEL = None

    @classmethod
    def load(
        cls,
        high_noise_model_path: str,
        low_noise_model_path: str = None,
        run_dtype="float16",
        rms_dtype="float",
        linear_backend: KsanaLinearBackend | str = KsanaLinearBackend.DEFAULT,
        attention_config: KsanaAttentionConfig | None = None,
        model_boundary=None,
        torch_compile_args=None,
        lora=None,
        comfy_progress_bar_func=None,
    ):
        # Qwen-Image is much more stable in bfloat16; fp16 frequently overflows to NaN in practice.
        # Our own example script uses bfloat16 by default.
        if "qwen" in high_noise_model_path.lower() and "image" in high_noise_model_path.lower():
            if run_dtype in ("float16", "fp16", "torch.float16"):
                log.warning("qwen-image detected: forcing run_dtype to bfloat16 for numerical stability.")
                run_dtype = "bfloat16"

        num_gpus = get_gpu_count()
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
            rms_dtype=rms_dtype,
            linear_backend=KsanaLinearBackend(linear_backend),
            attention_config=KsanaAttentionConfig() if attention_config is None else attention_config,
            torch_compile_config=torch_compile_args,
            boundary=model_boundary,
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
        if not high_noise_model_path:
            raise ValueError("high_noise_model_path is empty; check ComfyUI diffusion_models paths.")
        if not os.path.exists(high_noise_model_path):
            raise FileNotFoundError(f"high_noise_model_path not found: {high_noise_model_path}")
        if low_noise_model_path is not None and not os.path.exists(low_noise_model_path):
            raise FileNotFoundError(f"low_noise_model_path not found: {low_noise_model_path}")

        ksana_engine = get_engine(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            ksana_engine.clear_models(cls.LOADED_MODEL)

        try:
            loaded_model_keys = ksana_engine.load_diffusion_model(
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
            cls.LOADED_MODEL = None
            log.exception("load_diffusion_model failed")
            raise RuntimeError(
                f"load_diffusion_model failed for: {high_noise_model_path} ({type(e).__name__}: {e})"
            ) from e

        cls.LOADED_MODEL = loaded_model_keys
        MemoryProfiler.record_memory("after_load_model")
        return KsanaNodeModelLoaderOutput(
            model=cls.LOADED_MODEL,
            run_dtype=model_config.run_dtype,
        )
