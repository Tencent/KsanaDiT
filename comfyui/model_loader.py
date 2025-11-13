# adapt from comfyui/sd.py
import comfy
import folder_paths
from comfy import model_detection
import logging
import torch
import copy
import comfy.model_management as mm

from kdit.utils import get_gpu_count, time_range
from kdit import create_kdit_model

from comfy.model_patcher import ModelPatcher


class CustomModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        self.lora_cache = {}

    def clone(self):
        n = CustomModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.lora_cache = copy.copy(self.lora_cache)
        assert False, "not clone"
        return n

    def partially_unload(self, device_to, memory_to_free=0):
        with self.use_ejected():
            # unload
            # TODO: maybe control inside kdit_generator, and do not need CustomModelPatcher
            self.model.kdit_generator.to_cpu()

            self.model.device = torch.device("cpu")
            memory_freed = self.model.model_loaded_weight_memory
            self.model.model_loaded_weight_memory = 0
            assert False, "not partially_unload"
            return memory_freed

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        with self.use_ejected():
            self.unpatch_hooks()
            # load
            self.model.kdit_generator.to_gpu()

            # TODO: should be device_to?
            self.model.device = torch.device("cuda:0")
            self.model.model_loaded_weight_memory = self.size
            self.apply_hooks(self.forced_hooks, force_apply=True)
            assert False, "not load"


def load_diffusion_model_state_dict(
    model_path, model_options={}, load_ori_weights=False
):  # load unet in diffusers or regular format
    sd = comfy.utils.load_torch_file(model_path)

    dtype = model_options.get("dtype", None)

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
    print(
        f"input dtype: {dtype}, weight_dtype: {weight_dtype}, supported_dtype: {unet_weight_dtype}, unet_dtype: {unet_dtype}, manual_cast_dtype: {manual_cast_dtype}"
    )

    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", None)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model_config.unet_config["disable_unet_model_creation"] = True
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)

    sdkeys = list(sd.keys())
    newsdkeys = list(new_sd.keys())
    print(f"sd keys samples: {len(sdkeys)}, new_sd keys samples: {len(newsdkeys)}")
    print(f"Load_device: {load_device}, Offload_device: {offload_device}")
    print(
        f"unet_config: {model_config.unet_config}, latent_format: {model_config.latent_format}, model_options:{model_options}"
    )

    # Add Executor Logic
    unet_config = model_config.unet_config
    del unet_config["disable_unet_model_creation"]
    manual_cast_dtype = model_config.manual_cast_dtype
    if model_config.custom_operations is None:
        operations = comfy.ops.pick_operations(unet_config.get("dtype", None), manual_cast_dtype)
    else:
        operations = model_config.custom_operations
    print(f"custom_operations: {model_config.custom_operations}, operations:{operations}")
    print(
        f"unet_config: {model_config.unet_config}, model_options:{model_options}, diffusion_model_prefix:{diffusion_model_prefix}"
    )

    kdit_model = create_kdit_model(model_path, unet_config)
    kdit_model.load(
        model_path,
        comfy_model_config=unet_config,
        comfy_model_state_dict=new_sd,
        comfy_model_options=model_options,
        disable_weight_init_operations=operations,
        dtype=unet_dtype,
        load_device=load_device,
        offload_device=offload_device,
    )
    model.kdit_model = kdit_model

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
def load_diffusion_model(model_path, model_options={}):
    model = load_diffusion_model_state_dict(model_path, model_options=model_options, load_ori_weights=False)

    if model is None:
        logging.error("ERROR UNSUPPORTED MODEL {}".format(model_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(model_path))
    return model


class kDitModelLoaderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    [
                        "default",
                        "fp32",
                        "bf16",
                        "fp16",
                        "fp16_fast",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                    ],
                    {"default": "default"},
                ),
            },
            "optional": {
                "compile_args": ("STRING", {"default": "disabled"}),
            },
        }

    RETURN_TYPES = ("KDITMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "kdit"

    @classmethod
    def VALIDATE_INPUTS(s, model_name):
        return True

    def load_model(self, model_name, weight_dtype, compile_args: str = None):
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        # elif weight_dtype == "bf16":
        #     model_options["dtype"] = torch.bfloat16

        num_gpus = get_gpu_count()
        if num_gpus != 1:
            raise RuntimeError(f"only one GPU supported yet, but got {num_gpus}")

        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        print(f"Start to load diffusion model {model_name}: {model_path} with {num_gpus} gpus")

        model = load_diffusion_model(model_path, model_options=model_options)
        return (model,)
