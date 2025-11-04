# adapt from comfyui/sd.py
import comfy
import folder_paths
from comfy import model_detection
import logging
import torch
import copy
import comfy.model_management as mm

from vdit.utils import get_gpu_count
from vdit import create_vdit_model

from comfy.model_patcher import ModelPatcher
class CustomModelPatcher(ModelPatcher):
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        self.lora_cache = {}

    def clone(self):
        n = CustomModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
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
            #unload
            #TODO: maybe control inside vdit_generator, and do not need CustomModelPatcher
            self.model.vdit_generator.to_cpu()
            
            self.model.device = torch.device('cpu')
            memory_freed = self.model.model_loaded_weight_memory
            self.model.model_loaded_weight_memory = 0
            assert False, "not partially_unload"
            return memory_freed

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            self.unpatch_hooks()
            #load
            self.model.vdit_generator.to_gpu()

            # TODO: should be device_to?
            self.model.device = torch.device('cuda:0')
            self.model.model_loaded_weight_memory = self.size
            self.apply_hooks(self.forced_hooks, force_apply=True)
            assert False, "not load"
            


def load_diffusion_model_state_dict(model_path, model_options={}, load_ori_weights=False): #load unet in diffusers or regular format
    sd = comfy.utils.load_torch_file(model_path)
    
    dtype = model_options.get("dtype", None)

    #Allow loading unets from checkpoint files
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
        if new_sd is not None: #diffusers mmdit
            model_config = model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else: #diffusers unet
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
        unet_dtype = mm.unet_dtype(model_params=parameters, supported_dtypes=unet_weight_dtype, weight_dtype=weight_dtype)
    else:
        unet_dtype = dtype
    manual_cast_dtype = mm.unet_manual_cast(unet_dtype, load_device, model_config.supported_inference_dtypes)
    print(f"input dtype: {dtype}, weight_dtype: {weight_dtype}, supported_dtype: {unet_weight_dtype}, unet_dtype: {unet_dtype}, manual_cast_dtype: {manual_cast_dtype}")

    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", None)
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True
        
    model_config.unet_config["disable_unet_model_creation"] = True
    model = model_config.get_model(new_sd, "")
    model = model.to(offload_device)

    sdkeys = list(sd.keys())
    logging.info(f"sd keys samples: {len(sdkeys)} {sdkeys[0]}, {sdkeys[10]}")
    # # print_recursive(sd)
    newsdkeys = list(new_sd.keys())
    logging.info(f"new_sd keys samples: {len(newsdkeys)} {newsdkeys[0]}, {newsdkeys[10]}")
    logging.info(f"Load_device: {load_device}, Offload_device: {offload_device}")
    logging.info(f"unet_config: {model_config.unet_config}, model_options:{model_options}")
    
    # Add Executor Logic
    unet_config = model_config.unet_config
    del unet_config["disable_unet_model_creation"]
    manual_cast_dtype = model_config.manual_cast_dtype
    if model_config.custom_operations is None:
        operations = comfy.ops.pick_operations(unet_config.get("dtype", None), manual_cast_dtype)
    else:
        operations = model_config.custom_operations
        
        
    logging.info(f"unet_config: {model_config.unet_config}, model_options:{model_options}, diffusion_model_prefix:{diffusion_model_prefix}")
        
    vdit_model = create_vdit_model(model_path, unet_config)
    vdit_model.load(model_path, comfy_model_config=unet_config, comfy_model_state_dict=new_sd, comfy_model_options=model_options, load_device=load_device, offload_device=offload_device)
    model.vdit_model = vdit_model
    
    if load_ori_weights:
        model.load_model_weights(new_sd, "")
        left_over = sd.keys()
        if len(left_over) > 0:
            logging.info("left over keys in unet: {}".format(left_over))

    
    def sd_size(sd):
        module_mem = 0
        for k in sd:
            t = sd[k]
            module_mem += t.nelement() * torch.tensor([], dtype=unet_dtype).element_size()
        return module_mem

    return CustomModelPatcher(model, load_device=load_device, offload_device=offload_device, size=sd_size(sd))

def load_diffusion_model(model_path, model_options={}):
    
    model = load_diffusion_model_state_dict(model_path, model_options=model_options, load_ori_weights=False)
    
    if model is None:
        logging.error("ERROR UNSUPPORTED MODEL {}".format(model_path))
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(model_path))
    return model

# class WanVideoModel(comfy.model_base.BaseModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pipeline = {}

#     def __getitem__(self, k):
#         return self.pipeline[k]

#     def __setitem__(self, k, v):
#         self.pipeline[k] = v

# try:
#     from comfy.latent_formats import Wan21, Wan22
#     latent_format = Wan21
# except: #for backwards compatibility
#     log.warning("WARNING: Wan21 latent format not found, update ComfyUI for better live video preview")
#     from comfy.latent_formats import HunyuanVideo
#     latent_format = HunyuanVideo


# class WanVideoModelConfig:
#     def __init__(self, dtype, latent_format=latent_format):
#         self.unet_config = {}
#         self.unet_extra_config = {}
#         self.latent_format = latent_format
#         #self.latent_format.latent_channels = 16
#         self.manual_cast_dtype = dtype
#         self.sampling_settings = {"multiplier": 1.0}
#         self.memory_usage_factor = 2.0
#         self.unet_config["disable_unet_model_creation"] = True

class vDitModelLoaderNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (["default", "fp32", "bf16", "fp16", "fp16_fast", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"default": "default"}),
            },
            "optional": {
                "compile_args": ("STRING", {"default": "disabled"}),
            }
        }
    RETURN_TYPES = ("VDITMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "vdit"

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
        logging.info(f'Start to load diffusion model {model_name}: {model_path} with {num_gpus} gpus')
        
        model = load_diffusion_model(model_path, model_options=model_options)
        return (model,)


# class VAEDecode:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
#                 "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."})
#             }
#         }
#     RETURN_TYPES = ("IMAGE",)
#     OUTPUT_TOOLTIPS = ("The decoded image.",)
#     FUNCTION = "decode"

#     CATEGORY = "latent"
#     DESCRIPTION = "Decodes latent images back into pixel space images."

#     def decode(self, vae, samples):
#         images = vae.decode(samples["samples"])
#         if len(images.shape) == 5: #Combine batches
#             images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
#         return (images, )
    
# SCHEDULER_HANDLERS = {
#     "simple": SchedulerHandler(simple_scheduler),
#     "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
#     "karras": SchedulerHandler(k_diffusion_sampling.get_sigmas_karras, use_ms=False),
#     "exponential": SchedulerHandler(k_diffusion_sampling.get_sigmas_exponential, use_ms=False),
#     "ddim_uniform": SchedulerHandler(ddim_scheduler),
#     "beta": SchedulerHandler(beta_scheduler),
#     "normal": SchedulerHandler(normal_scheduler),
#     "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
#     "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
# }
# SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)

# KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
#                   "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
#                   "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
#                   "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
#                   "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece"]


# SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

# class KSAMPLER(Sampler):
#     def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
#         self.sampler_function = sampler_function
#         self.extra_options = extra_options
#         self.inpaint_options = inpaint_options

#     def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
#         extra_args["denoise_mask"] = denoise_mask
#         model_k = KSamplerX0Inpaint(model_wrap, sigmas)
#         model_k.latent_image = latent_image
#         if self.inpaint_options.get("random", False): #TODO: Should this be the default?
#             generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
#             model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
#         else:
#             model_k.noise = noise

#         noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

#         k_callback = None
#         total_steps = len(sigmas) - 1
#         if callback is not None:
#             k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

#         samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
#         samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
#         return samples
    

# def ksampler(sampler_name, extra_options={}, inpaint_options={}):
#     if sampler_name == "dpm_fast":
#         def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
#             if len(sigmas) <= 1:
#                 return noise

#             sigma_min = sigmas[-1]
#             if sigma_min == 0:
#                 sigma_min = sigmas[-2]
#             total_steps = len(sigmas) - 1
#             return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
#         sampler_function = dpm_fast_function
#     elif sampler_name == "dpm_adaptive":
#         def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
#             if len(sigmas) <= 1:
#                 return noise

#             sigma_min = sigmas[-1]
#             if sigma_min == 0:
#                 sigma_min = sigmas[-2]
#             return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
#         sampler_function = dpm_adaptive_function
#     else:
#         sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

#     return KSAMPLER(sampler_function, extra_options, inpaint_options)

# def sampler_object(name):
#     if name == "uni_pc":
#         sampler = KSAMPLER(uni_pc.sample_unipc)
#     elif name == "uni_pc_bh2":
#         sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
#     elif name == "ddim":
#         sampler = ksampler("euler", inpaint_options={"random": True})
#     else:
#         sampler = ksampler(name)
#     return sampler


# def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
#     cfg_guider = CFGGuider(model)
#     cfg_guider.set_conds(positive, negative)
#     cfg_guider.set_cfg(cfg)
#     return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


# class KSampler:
#     SCHEDULERS = SCHEDULER_NAMES
#     SAMPLERS = SAMPLER_NAMES
#     DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

#     def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
#         self.model = model
#         self.device = device
#         if scheduler not in self.SCHEDULERS:
#             scheduler = self.SCHEDULERS[0]
#         if sampler not in self.SAMPLERS:
#             sampler = self.SAMPLERS[0]
#         self.scheduler = scheduler
#         self.sampler = sampler
#         self.set_steps(steps, denoise)
#         self.denoise = denoise
#         self.model_options = model_options

#     def calculate_sigmas(self, steps):
#         sigmas = None

#         discard_penultimate_sigma = False
#         if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
#             steps += 1
#             discard_penultimate_sigma = True

#         sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

#         if discard_penultimate_sigma:
#             sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
#         return sigmas

#     def set_steps(self, steps, denoise=None):
#         self.steps = steps
#         if denoise is None or denoise > 0.9999:
#             self.sigmas = self.calculate_sigmas(steps).to(self.device)
#         else:
#             if denoise <= 0.0:
#                 self.sigmas = torch.FloatTensor([])
#             else:
#                 new_steps = int(steps/denoise)
#                 sigmas = self.calculate_sigmas(new_steps).to(self.device)
#                 self.sigmas = sigmas[-(steps + 1):]

#     def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
#         if sigmas is None:
#             sigmas = self.sigmas

#         if last_step is not None and last_step < (len(sigmas) - 1):
#             sigmas = sigmas[:last_step + 1]
#             if force_full_denoise:
#                 sigmas[-1] = 0

#         if start_step is not None:
#             if start_step < (len(sigmas) - 1):
#                 sigmas = sigmas[start_step:]
#             else:
#                 if latent_image is not None:
#                     return latent_image
#                 else:
#                     return torch.zeros_like(noise)

#         sampler = sampler_object(self.sampler)

#         return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


# def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
#                     latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
#     latent_image = latent["samples"]
#     latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

#     if disable_noise:
#         noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
#     else:
#         batch_inds = latent["batch_index"] if "batch_index" in latent else None
#         noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

#     noise_mask = None
#     if "noise_mask" in latent:
#         noise_mask = latent["noise_mask"]

#     callback = latent_preview.prepare_callback(model, steps)
#     disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
#     sampler = KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name,
#                                       scheduler=scheduler, denoise=denoise, model_options=model.model_options)

#     samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image,
#                              start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
#     samples = samples.to(comfy.mm.intermediate_device())

#     # samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
#     #                               denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
#     #                               force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
#     out = latent.copy()
#     out["samples"] = samples
#     return (out, )

# class KSamplerAdvanced:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"model": ("MODEL",),
#                     "add_noise": (["enable", "disable"], ),
#                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
#                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
#                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
#                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
#                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
#                     "positive": ("CONDITIONING", ),
#                     "negative": ("CONDITIONING", ),
#                     "latent_image": ("LATENT", ),
#                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
#                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
#                     "return_with_leftover_noise": (["disable", "enable"], ),
#                      }
#                 }

#     RETURN_TYPES = ("LATENT",)
#     FUNCTION = "sample"

#     CATEGORY = "sampling"

#     def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
#                positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
#         force_full_denoise = True
#         if return_with_leftover_noise == "enable":
#             force_full_denoise = False
#         disable_noise = False
#         if add_noise == "disable":
#             disable_noise = True


#         return common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


