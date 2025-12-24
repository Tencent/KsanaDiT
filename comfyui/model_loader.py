import folder_paths
import comfy.model_management as mm

from ksana.utils import get_gpu_count, time_range, log
from ksana import get_generator
from ksana.config import KsanaModelConfig, KsanaDistributedConfig
from ksana.utils.profile import MemoryProfiler

from comfy.utils import ProgressBar


@time_range
def load_comfy_model_from_name(
    model_name: list,
    num_gpus,
    ksana_model_config: KsanaModelConfig,
    comfy_bar_callback=None,
    lora=None,
):
    high, low = model_name
    high_model_path = folder_paths.get_full_path("diffusion_models", high)
    low_model_path = None
    if low is not None:
        low_model_path = folder_paths.get_full_path("diffusion_models", low)

    ksana_generator = get_generator(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
    ksana_model = ksana_generator.load_diffusion_model(
        model_path=(high_model_path, low_model_path) if low_model_path is not None else high_model_path,
        model_config=ksana_model_config,
        comfy_bar_callback=comfy_bar_callback,
        lora=lora,
    )
    return ksana_model


class KsanaModelLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": {
                # attn_backend dtype > linear_backend dtype > run_dtype
                "run_dtype": (
                    ["float16", "bfloat16"],
                    {"default": "float16"},
                    {"tooltip": "dtype of running model"},
                ),
                "linear_backend": (
                    ["default", "fp8_gemm", "fp8_gemm_dynamic", "fp16_gemm"],
                    {"default": "default"},
                    {"tooltip": "linear_backend default use linear dtype from model"},
                ),
                "attn_backend": (
                    ["flash_attention", "sage_attention"],
                    {"default": "flash_attention"},
                    {"tooltip": "attention backend"},
                ),
                "low_noise_model_name": (
                    folder_paths.get_filename_list("diffusion_models") + ["Empty"],
                    {"default": "Empty"},
                ),
                "model_boundary": (
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
                "num_gpus": (["default", "1"], {"default": "default"}),
                "torch_compile_args": ("KSANACOMPILEARGS", {"default": None}),
                "lora": ("KSANALORA", {"default": None}),
            },
        }

    RETURN_TYPES = ("KSANAMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ksana"

    @classmethod
    def VALIDATE_INPUTS(s):
        return True

    def load_model(
        self,
        model_name,
        run_dtype="float16",
        linear_backend="default",
        attn_backend="default",
        num_gpus="default",
        low_noise_model_name="Empty",
        model_boundary=None,
        torch_compile_args=None,
        lora=None,
    ):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        num_gpus = get_gpu_count() if num_gpus == "default" else int(num_gpus)

        model_config = KsanaModelConfig(
            run_dtype=run_dtype,
            linear_backend=linear_backend,
            attn_backend=attn_backend,
            torch_compile_config=torch_compile_args,
        )
        comfyui_progress_bar = ProgressBar(1 if low_noise_model_name == "Empty" else 2)

        def comfy_bar_callback():
            comfyui_progress_bar.update(1)

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

        MemoryProfiler.record_memory(f"before_load_{model_name}, {low_noise_model_name}")
        model = load_comfy_model_from_name(
            [model_name, low_noise_model_name],
            num_gpus=num_gpus,
            ksana_model_config=model_config,
            comfy_bar_callback=comfy_bar_callback,
            lora=[high_model_loras_list, low_model_loras_list],
        )
        MemoryProfiler.record_memory(f"after_load_{model_name}, {low_noise_model_name}")
        return (
            {
                "model": model,
                "model_name": model_name,  # TODO:  need remove
                "run_dtype": model_config.run_dtype,
                "boundary": model_boundary,
            },
        )
