import comfy
import folder_paths
from comfy import model_detection
import logging
import torch
import comfy.model_management as mm

from ksana.utils import get_gpu_count, time_range
from ksana import get_generator
from ksana.config import KsanaModelConfig, KsanaDistributedConfig
from ksana.utils.profile import MemoryProfiler

from comfy.model_patcher import ModelPatcher


class CustomModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)


def load_diffusion_model_state_dict(
    model_path, num_gpus, ksana_model_config: KsanaModelConfig, load_ori_weights=False
):  # load unet in diffusers or regular format
    sd = comfy.utils.load_torch_file(model_path)

    dtype = None

    # Allow loading unets from checkpoint files
    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd

    parameters = comfy.utils.calculate_parameters(sd)
    sd_dtype = comfy.utils.weight_dtype(sd)

    load_device = mm.get_torch_device()
    model_config = model_detection.model_config_from_unet(sd, "")

    if model_config is not None:
        new_sd = sd
    else:
        new_sd = model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:  # diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:  # diffusers unet
            model_config = model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)

            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning("{} {}".format(diffusers_keys[k], k))

    offload_device = mm.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.scaled_fp8 is not None:
        sd_dtype = None

    if dtype is None:
        unet_dtype = mm.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=sd_dtype,
        )
    else:
        unet_dtype = dtype
    manual_cast_dtype = mm.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    assert (
        manual_cast_dtype is None or manual_cast_dtype == unet_dtype
    ), "manual_cast_dtype must be the same as unet_dtype"
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = None

    model_config.unet_config["disable_unet_model_creation"] = True
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)

    sdkeys = list(sd.keys())
    newsdkeys = list(new_sd.keys())

    # Add Executor Logic
    unet_config = model_config.unet_config
    del unet_config["disable_unet_model_creation"]

    # set ksana_model_config
    ksana_model_config.run_dtype = unet_dtype
    scaled_fp8_dtype = model_config.scaled_fp8
    if (
        ksana_model_config.linear_backend == "default"
        and scaled_fp8_dtype is not None
        and "float8" in str(scaled_fp8_dtype)
    ):
        print("linear_backend will use fp8_gemm")
        ksana_model_config.linear_backend = "fp8_gemm"

    print(
        f"input dtype: {dtype}, sd_weight_dtype: {sd_dtype}, supported_dtype: {unet_weight_dtype}, unet_dtype: {unet_dtype}, "
        f"manual_cast_dtype: {manual_cast_dtype}, scaled_fp8: {scaled_fp8_dtype}"
    )
    print(f"sd keys samples: {len(sdkeys)}, new_sd keys samples: {len(newsdkeys)}")
    print(f"Load_device: {load_device}, Offload_device: {offload_device}")
    print(
        f"unet_config: {model_config.unet_config}, latent_format: {model_config.latent_format}, ksana_model_config:{ksana_model_config}, diffusion_model_prefix:{diffusion_model_prefix}"
    )

    ksana_generator = get_generator(dist_config=KsanaDistributedConfig(num_gpus=num_gpus))
    ksana_model = ksana_generator.load_diffusion_model(
        model_path=model_path,
        model_config=ksana_model_config,
        comfy_model_config=unet_config,
        comfy_model_state_dict=new_sd,
    )
    model.ksana_model = ksana_model

    if load_ori_weights:
        model.comfy_load_model_weights(new_sd, "")
        left_over = sd.keys()
        if len(left_over) > 0:
            print("left over keys in unet: {}".format(left_over))

    def sd_size(sd):
        module_mem = 0
        for k in sd:
            t = sd[k]
            module_mem += t.nelement() * torch.tensor([], dtype=unet_dtype).element_size()
        return module_mem

    return CustomModelPatcher(model, load_device=load_device, offload_device=offload_device, size=sd_size(sd))


@time_range
def load_diffusion_model(model_name, num_gpus, ksana_model_config: KsanaModelConfig):
    model_path = folder_paths.get_full_path("diffusion_models", model_name)
    model = load_diffusion_model_state_dict(
        model_path, num_gpus=num_gpus, ksana_model_config=ksana_model_config, load_ori_weights=False
    )

    if model is None:
        logging.error("ERROR UNSUPPORTED MODEL {}".format(model_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(model_path))
    return model


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
                    ["default", "fp8_gemm", "fp16_gemm"],
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
    ):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if linear_backend == "fp8_gemm":
            # TODO: run_dtype do not have default
            run_dtype_has_fp8 = "float8" in run_dtype.lower() or "fp8" in run_dtype.lower()
            if run_dtype != "default" and (not run_dtype_has_fp8):
                logging.warning(
                    f" run_dtype {run_dtype} can not use fp8_gemm linear_backend, will use run_dtype back to default"
                )
                run_dtype = "float16"
                linear_backend = "default"

        num_gpus = get_gpu_count() if num_gpus == "default" else int(num_gpus)

        model_config = KsanaModelConfig(
            run_dtype=run_dtype,
            linear_backend=linear_backend,
            attn_backend=attn_backend,
            torch_compile_config=torch_compile_args,
        )

        MemoryProfiler.record_memory(f"before_load_{model_name}")
        high_model = load_diffusion_model(model_name, num_gpus=num_gpus, ksana_model_config=model_config)
        MemoryProfiler.record_memory(f"after_load_{model_name}")
        low_model = None
        if low_noise_model_name is not None and low_noise_model_name != "Empty":
            MemoryProfiler.record_memory(f"before_load_{low_noise_model_name}")
            low_model = load_diffusion_model(low_noise_model_name, num_gpus=num_gpus, ksana_model_config=model_config)
            MemoryProfiler.record_memory(f"after_load_{low_noise_model_name}")
        print(f"high_model: {high_model}, low_model: {low_model}")
        return (
            {
                "high_noise_model": high_model,
                "low_noise_model": low_model,
                "boundary": model_boundary,
            },
        )
