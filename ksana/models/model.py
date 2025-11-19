import os
import torch
from abc import ABC
from ..utils import log, time_range, load_and_merge_lora_weight_from_safetensors, model_safe_downcast  # , ksanaProfiler
from .wan import WanModel, T5EncoderModel, Wan2_1_VAE, Wan2_2_VAE
from .wan.configs import WAN2_2_CONFIGS
import time


def get_default_model_config(model_name, model_type, model_size):
    if model_name == "wan2.2":
        config_type = f"{model_type}-{model_size}"
        return WAN2_2_CONFIGS[config_type]
    else:
        raise ValueError(f"model_name {model_name} not supported")


def create_ksana_model(model_path, comfy_model_config=None):
    full_model_name = os.path.basename(model_path)
    model_name, model_type, model_size = KsanaModel.get_model_type(
        full_model_name, comfy_model_config
    )  # wan2.2, t2v, A14B
    assert model_name in ["wan2.2"], "only support wan2.2 yet"
    model_config = get_default_model_config(model_name, model_type, model_size)
    return KsanaModel(model_config)


class KsanaT5Encoder(ABC):
    def __init__(self, model_config, checkpoint_dir, shard_fn):
        _default_config = model_config
        self.model = T5EncoderModel(
            text_len=_default_config.text_len,
            dtype=_default_config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, _default_config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, _default_config.t5_tokenizer),
            shard_fn=shard_fn,
        )

    def forward(self, text):
        # TODO: or other device
        return self.model(text, device=torch.device("cpu"))

    def to(self, device):
        self.model.model.to(device)


class KsanaVAE(ABC):
    def __init__(self, vae_type, model_config, checkpoint_dir, device, dtype=torch.float):
        _default_config = model_config
        if vae_type == "wan2_1":
            self.model = Wan2_1_VAE(
                vae_pth=os.path.join(checkpoint_dir, _default_config.vae_checkpoint),
                dtype=dtype,
                device=device,
            )
        elif vae_type == "vae_2.2":
            self.model = Wan2_2_VAE(
                vae_pth=os.path.join(checkpoint_dir, _default_config.vae_checkpoint),
                dtype=dtype,
                device=device,
            )
        else:
            raise ValueError(f"model_name {self.model_name} not supported")

        self.z_dim = self.model.model.z_dim
        self.vae_stride = _default_config.vae_stride
        self.patch_size = _default_config.patch_size

    def decode(self, latents):
        return self.model.decode(latents)


class KsanaModel(ABC):
    def __init__(self, model_config):
        self._default_model_config = model_config
        self._run_dtype = model_config.get("param_dtype", torch.float16)

    @time_range
    def load(
        self,
        comfy_model_path: str = None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_model_options: dict = None,
        disable_weight_init_operations=None,
        dtype=None,
        load_device=None,
        offload_device=None,
        checkpoint_dir=None,
        subfolder=None,
        lora_dir=None,
        torch_compile_config=None,
    ):
        assert self.model_name in ["wan2.2"], "only support wan2.2 yet"
        if "wan" in self.model_name:
            self.load_wan_model(
                comfy_model_path=comfy_model_path,
                comfy_model_config=comfy_model_config,
                comfy_model_state_dict=comfy_model_state_dict,
                comfy_model_options=comfy_model_options,
                disable_weight_init_operations=disable_weight_init_operations,
                dtype=dtype,
                load_device=load_device,
                offload_device=offload_device,
                checkpoint_dir=checkpoint_dir,
                subfolder=subfolder,
                lora_dir=lora_dir,
                torch_compile_config=torch_compile_config,
            )
        else:
            raise ValueError(f"model_name {self.model_name} not supported")

    def apply_torch_compile(self, torch_compile_config=None):
        if torch_compile_config is None:
            return
        log.info(f"apply torch_compile_config: {torch_compile_config}")
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
            torch._dynamo.config.cache_size_limit = torch_compile_config.dynamo_cache_size_limit
            torch._dynamo.config.force_parameter_static_shapes = torch_compile_config.force_parameter_static_shapes
            try:
                torch._dynamo.config.recompile_limit = torch_compile_config.dynamo_recompile_limit
            except Exception as e:
                log.warning(f"Could not set recompile_limit: {e}")
        if torch_compile_config.compile_transformer_blocks_only:
            log.info("Compiling only transformer blocks")
            compiled_cnt = 0
            for i, block in enumerate(self.model.blocks):
                try:
                    self.model.blocks[i] = torch.compile(
                        block,
                        backend=torch_compile_config.backend,
                        mode=torch_compile_config.mode,
                        fullgraph=torch_compile_config.fullgraph,
                        dynamic=torch_compile_config.dynamic,
                    )
                    compiled_cnt += 1
                except Exception as e:
                    log.warning(f"torch.compile block[{i}] failed: {e}")
            log.info(f"Applied torch.compile to {compiled_cnt}/{len(self.model.blocks)} transformer blocks.")
        else:
            log.info("Compiling entire model")
            self.model = torch.compile(
                self.model,
                fullgraph=torch_compile_config.fullgraph,
                dynamic=torch_compile_config.dynamic,
                backend=torch_compile_config.backend,
                mode=torch_compile_config.mode,
            )

    def load_model_weights(self, model, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = model.model_config.process_unet_state_dict(to_load)
        m, u = model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            log.warning("unet missing: {}".format(m))

        if len(u) > 0:
            log.warning("unet unexpected: {}".format(u))
        del to_load
        return model

    def load_wan_model(
        self,
        comfy_model_path: str = None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_model_options: dict = None,
        disable_weight_init_operations=None,
        dtype=None,
        load_device=None,
        offload_device=None,
        checkpoint_dir=None,
        subfolder=None,
        lora_dir=None,
        torch_compile_config=None,
    ):
        # with ksanaProfiler("KsanaModel.load_wan_model"):
        if comfy_model_config is not None and comfy_model_state_dict is not None:
            log.info(
                f"load from comfy_model_path:{comfy_model_path}, comfy_model_config:{comfy_model_config}, comfy_model_options:{comfy_model_options}, "
                f"disable_weight_init_operations:{disable_weight_init_operations}, dtype:{dtype}, load_device:{load_device}, offload_device:{offload_device}"
            )
            in_dim = comfy_model_config.get("in_dim", self.default_model_config.get("in_dim", 16))
            out_dim = comfy_model_config.get("out_dim", self.default_model_config.get("out_dim", 16))
            self.set_run_dtype(dtype)
            start = time.time()
            self.model = WanModel(
                model_type=self.task_type,
                patch_size=self.default_model_config.patch_size,
                text_len=self.default_model_config.text_len,
                in_dim=in_dim,
                dim=self.default_model_config.dim,
                ffn_dim=self.default_model_config.ffn_dim,
                freq_dim=self.default_model_config.freq_dim,
                out_dim=out_dim,
                num_heads=self.default_model_config.num_heads,
                num_layers=self.default_model_config.num_layers,
                window_size=self.default_model_config.window_size,
                qk_norm=self.default_model_config.qk_norm,
                cross_attn_norm=self.default_model_config.cross_attn_norm,
                eps=self.default_model_config.eps,
                disable_weight_init_operations=disable_weight_init_operations,
                device=offload_device,
                dtype=dtype,
            )
            stop1 = time.time()
            self.model.load_state_dict(comfy_model_state_dict, strict=False)
            stop2 = time.time()
            log.info(f"create model takes: {(stop1 - start):.2f}, load states takes {(stop2 - stop1):.2f} seconds")
        else:
            log.info(f"load from checkpoint_dir:{checkpoint_dir}, subfolder:{subfolder}")
            self.model = WanModel.from_pretrained(checkpoint_dir, subfolder=subfolder)

        log.debug(f"model: {self.model}")
        self.apply_torch_compile(torch_compile_config)

        # self.high_noise_model = self._configure_model(
        #     model=self.high_noise_model,
        #     use_sp=use_sp,
        #     dit_fsdp=dit_fsdp,
        #     shard_fn=shard_fn,
        #     convert_model_dtype=convert_model_dtype)
        if lora_dir:
            log.info(f"load lora from {lora_dir}")
            self.model.set_keep_in_fp32_modules()
            self.model = load_and_merge_lora_weight_from_safetensors(self.model, lora_dir)
            model_safe_downcast(
                self.model,
                dtype=self.run_dtype,
                keep_in_fp32_modules=[],
                keep_in_fp32_parameters=self.model._keep_in_fp32_params,
            )

        # def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
        #                      convert_model_dtype):
        #     """
        #     Configures a model object. This includes setting evaluation modes,
        #     applying distributed parallel strategy, and handling device placement.

        #     Args:
        #         model (torch.nn.Module):
        #             The model instance to configure.
        #         use_sp (`bool`):
        #             Enable distribution strategy of sequence parallel.
        #         dit_fsdp (`bool`):
        #             Enable FSDP sharding for DiT model.
        #         shard_fn (callable):
        #             The function to apply FSDP sharding.
        #         convert_model_dtype (`bool`):
        #             Convert DiT model parameters dtype to 'config.param_dtype'.
        #             Only works without FSDP.

        #     Returns:
        #         torch.nn.Module:
        #             The configured model.
        #     """
        #     model.eval().requires_grad_(False)

        #     if use_sp:
        #         for block in model.blocks:
        #             block.self_attn.forward = types.MethodType(
        #                 sp_attn_forward, block.self_attn)
        #         model.forward = types.MethodType(sp_dit_forward, model)

        #     if dist.is_initialized():
        #         dist.barrier()

        #     if dit_fsdp:
        #         model = shard_fn(model)
        #     else:
        #         if convert_model_dtype:
        #             model.to(self.param_dtype)
        #         if not self.init_on_cpu:
        #             model.to(self.device)

        #     return model

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    @property
    def model_name(self):
        return self._default_model_config.model_name

    @property
    def task_type(self):
        return self._default_model_config.task_type

    @property
    def model_size(self):
        return self._default_model_config.model_size

    @property
    def default_model_config(self):
        return self._default_model_config

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def run_dtype(self):
        return self._run_dtype

    def set_run_dtype(self, dtype):
        self._run_dtype = dtype

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_model_type(full_model_name: str, comfy_model_config: dict = None):
        """
        Get the model type from the state dict.
        model_name: wan2.2, wan2.1
        model_type: t2v or s2v or i2v or ti2v
        model_size: A14B, 5B. etc.
        return (model_name, model_type, model_size)
        """
        choise = ["t2v", "s2v", "i2v", "ti2v"]
        model_type = None if comfy_model_config is None else comfy_model_config.get("model_type", None)
        lower_model_name = full_model_name.lower()
        if model_type is not None:
            if model_type not in choise:
                raise ValueError(f"model_type {model_type} not in {choise}")
        else:
            for t in choise:
                if t in lower_model_name:
                    model_type = t
                    break
        model_name = None
        wan22 = ["wan2.2", "wan22", "wan2_2"]
        for k in wan22:
            if k in lower_model_name:
                model_name = "wan2.2"
                break

        model_size = None
        if model_name == "wan2.2":
            if "14b" in lower_model_name:
                model_size = "A14B"
            elif "5b" in lower_model_name:
                model_size = "5B"
        elif model_name == "wan2.1":
            if "14b" in lower_model_name:
                model_size = "14B"
            elif "1.3b" in lower_model_name or "1_3b" in lower_model_name:
                model_size = "1.3B"
        else:
            raise ValueError(f"can not detect model_size from model_name:{model_name}, model_name:{model_name}")

        log.info(f"model_name:{model_name}, model_type: {model_type}, model_size: {model_size}")
        return model_name, model_type, model_size
