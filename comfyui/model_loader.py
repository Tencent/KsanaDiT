import comfy
import folder_paths
from comfy import model_detection
import logging
import torch
import comfy.model_management as mm

from ksana.utils import get_gpu_count, time_range
from ksana import get_generator
from ksana.config import KsanaModelConfig
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
    weight_dtype = comfy.utils.weight_dtype(sd)

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
        weight_dtype = None

    if dtype is None:
        unet_dtype = mm.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype
    manual_cast_dtype = mm.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = None
    linear_backend = ksana_model_config.linear_backend
    scaled_fp8_dtype = model_config.scaled_fp8
    if linear_backend == "default":
        model_config.optimizations["fp8"] = (
            True if scaled_fp8_dtype is not None and "float8" in str(scaled_fp8_dtype) else False
        )
    else:
        model_config.optimizations["fp8"] = True if linear_backend == "fp8_gemm" else False

    print(
        f"input dtype: {dtype}, weight_dtype: {weight_dtype}, supported_dtype: {unet_weight_dtype}, unet_dtype: {unet_dtype}, "
        f"manual_cast_dtype: {manual_cast_dtype}, scaled_fp8: {scaled_fp8_dtype}, linear_backend: {linear_backend}"
    )
    model_config.unet_config["disable_unet_model_creation"] = True
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)

    sdkeys = list(sd.keys())
    newsdkeys = list(new_sd.keys())
    print(f"sd keys samples: {len(sdkeys)}, new_sd keys samples: {len(newsdkeys)}")
    print(f"Load_device: {load_device}, Offload_device: {offload_device}")
    print(
        f"unet_config: {model_config.unet_config}, latent_format: {model_config.latent_format}, ksana_model_config:{ksana_model_config}"
    )

    # Add Executor Logic
    unet_config = model_config.unet_config
    del unet_config["disable_unet_model_creation"]
    manual_cast_dtype = model_config.manual_cast_dtype
    if model_config.custom_operations is None:
        fp8 = model_config.optimizations.get("fp8", False)
        operations = comfy.ops.pick_operations(
            unet_config.get("dtype", None), manual_cast_dtype, fp8_optimizations=fp8, scaled_fp8=scaled_fp8_dtype
        )
    else:
        operations = model_config.custom_operations
    print(f"custom_operations: {model_config.custom_operations}, operations:{operations}")
    print(f"unet_config: {model_config.unet_config}, diffusion_model_prefix:{diffusion_model_prefix}")

    if model_config.optimizations["fp8"]:
        ksana_model_config.linear_backend = "fp8_gemm"
    ksana_model_config.weight_dtype = unet_dtype

    ksana_generator = get_generator(num_gpus=num_gpus)
    ksana_model = ksana_generator.load_diffusion_model_from_comfy(
        model_config=ksana_model_config,
        comfy_model_path=model_path,
        comfy_model_config=unet_config,
        comfy_model_state_dict=new_sd,
        comfy_operations=operations,
        load_device=load_device,
        offload_device=offload_device,
    )
    model.ksana_model = ksana_model

    if load_ori_weights:
        model.load_model_weights(new_sd, "")
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
def load_diffusion_model(model_path, num_gpus, ksana_model_config: KsanaModelConfig):
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
                # attn_backend dtype > linear_backend dtye > weight_dtype
                "weight_dtype": (
                    ["default", "float16", "bfloat16"],
                    {"default": "default"},
                    {"tooltip": "weight dtype of running model"},
                ),
                "linear_backend": (
                    ["default", "fp8_gemm", "fp16_gemm"],
                    {"default": "default"},
                    {"tooltip": "linear_backend default use linear dtype from model"},
                ),
                "attn_backend": (
                    ["default", "flash_attention", "sage_attention"],
                    {"default": "flash_attention"},
                    {"tooltip": "attention backend"},
                ),
                "num_gpus": ("INT", {"default": 1}),
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
        weight_dtype="default",
        linear_backend="default",
        attn_backend="default",
        num_gpus=1,
        torch_compile_args=None,
    ):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if linear_backend == "fp8_gemm":
            weight_dtype_has_fp8 = "float8" in weight_dtype.lower() or "fp8" in weight_dtype.lower()
            if weight_dtype != "default" and (not weight_dtype_has_fp8):
                logging.warning(
                    f" weight_dtype {weight_dtype} can not use fp8_gemm linear_backend, will use weight_dtype back to default"
                )
                weight_dtype = "default"
                linear_backend = "default"

        if num_gpus > 1:
            if num_gpus > get_gpu_count():
                logging.warning(
                    f"num_gpus {num_gpus} is larger than gpu count {get_gpu_count()}, will use gpu count {get_gpu_count()}"
                )
                num_gpus = get_gpu_count()

        model_config = KsanaModelConfig(
            weight_dtype=weight_dtype,
            linear_backend=linear_backend,
            attn_backend=attn_backend,
            torch_compile_config=torch_compile_args,
        )

        num_gpus = get_gpu_count()
        if num_gpus != 1:
            raise RuntimeError(f"only one GPU supported yet, but got {num_gpus}")

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        print(f"Start to load diffusion model {model_name}: {model_path} with {num_gpus} gpus")
        MemoryProfiler.record_memory(f"before_load_{model_name}")
        model = load_diffusion_model(model_path, num_gpus=num_gpus, ksana_model_config=model_config)
        MemoryProfiler.record_memory(f"after_load_{model_name}")
        return (model,)
