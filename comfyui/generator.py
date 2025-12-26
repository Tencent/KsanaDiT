import comfy
import comfy.model_management as mm
from comfy.utils import ProgressBar
from ksana import get_generator
from ksana.config import KsanaSampleConfig, KsanaRuntimeConfig
from ksana.utils.profile import MemoryProfiler
from ksana.utils import log
from ksana.models.diffusion import KsanaDiffusionModel
from ksana.utils.memory import (
    estimate_ksana_model_memory,
    get_available_memory,
    MODEL_SIZE_MAP,
    MEMORY_USAGE_FACTOR_MAP,
)


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
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {
                        "default": "simple",
                        "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
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
                "rope_function": (
                    ["default", "comfy"],
                    {
                        "default": "default",
                        "tooltip": "Select the rotary positional embedding implementation.",
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
                "sigmas": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "run"
    CATEGORY = "ksana"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def _prepare_memory_for_ksana_models(self, model_name_str, latent_shape, run_dtype, device):
        try:
            # TODO：use model key
            model_name, task_type, model_size = KsanaDiffusionModel.get_model_type(model_name_str)

            model_weight_memory = MODEL_SIZE_MAP.get(model_size)
            if model_weight_memory is None:
                raise ValueError(f"Unknown model size: {model_size}")

            memory_usage_factor = MEMORY_USAGE_FACTOR_MAP.get((model_name, task_type, model_size), 1.0)

            total_memory_required, _ = estimate_ksana_model_memory(
                model_weight_memory, latent_shape, run_dtype, memory_usage_factor
            )
            available_memory = get_available_memory(device)
            if available_memory < total_memory_required:
                mm.free_memory(total_memory_required, device, keep_loaded=[])

            log.debug(f"Final free memory: {get_available_memory(device) / (1024*1024):.1f} MB")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare memory for KsanaDiT models: {e}")

    def run(
        self,
        model,
        positive,
        negative,
        latent_image,
        steps,
        seed,
        scheduler,
        solver_name,
        sample_guide_scale,
        sample_shift,
        denoise=1.0,
        rope_function="default",
        low_sample_guide_scale=None,
        high_cache_config=None,
        low_cache_config=None,
        sigmas=None,
    ):
        if sigmas is not None:
            expected_lengths = steps + 1
            if len(sigmas) != expected_lengths:
                raise RuntimeError(f"sigmas length ({len(sigmas)}) must be equal to steps + 1 ({expected_lengths})")

        ksana_model = model.get("model")
        run_dtype = model.get("run_dtype")
        boundary = model.get("boundary")
        ksana_generator = get_generator()
        MemoryProfiler.record_memory("before_ksana_generator_generate_video_with_tensors")

        latent_shape = latent_image["samples"].shape
        device = mm.get_torch_device()
        self._prepare_memory_for_ksana_models(
            model.get("model_name"), latent_shape=latent_shape, run_dtype=run_dtype, device=device
        )

        comfyui_progress_bar = ProgressBar(steps)

        def comfy_bar_callback(step, total):
            comfyui_progress_bar.update_absolute(step, total)

        num_prompts = positive[0][0].shape[0]
        batch_per_prompt = latent_image.get("batch_per_prompt", 1)
        batch_per_prompt = [batch_per_prompt] * num_prompts

        # TODO: maybe need to latent_format process_in for positive/negative?
        samples = ksana_generator.forward_diffusion_models_with_tensors(
            model_keys=ksana_model,
            positive=positive[0][0],
            negative=negative[0][0],
            img_latents=latent_image["samples"],  # [1, 16, 5, h/, w/]
            sample_config=KsanaSampleConfig(
                steps=steps,
                cfg_scale=(sample_guide_scale, low_sample_guide_scale),
                shift=sample_shift,
                solver=solver_name,
                denoise=denoise,
                sigmas=sigmas,
                batch_per_prompt=batch_per_prompt,
            ),
            runtime_config=KsanaRuntimeConfig(
                seed=seed,
                boundary=boundary,
                rope_function=rope_function,
            ),
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
            comfy_bar_callback=comfy_bar_callback,
        )
        MemoryProfiler.record_memory("after_ksana_generator_generate_video_with_tensors")
        if len(samples.shape) == 4:
            samples = samples.unsqueeze(0)

        return ({"samples": samples, "with_end_image": latent_image.get("with_end_image", False)},)
