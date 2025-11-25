import time
import types
from abc import ABC

import torch
import torch.distributed as dist

from ..distributed import sp_attn_forward, sp_dit_forward
from ..utils import (  # , ksanaProfiler
    load_and_merge_lora_weight_from_safetensors,
    log,
    model_safe_downcast,
    time_range,
)
from ..utils.const import DEFAULT_RUN_DTYPE
from .wan import WanModel
from .wan.configs import WAN2_2_CONFIGS
from ..config import KsanaModelConfig


class KsanaDiffusionModel(ABC):
    def __init__(self, model_config: KsanaModelConfig, pipeline_config, dist_config):
        self.model_name = pipeline_config.model_name
        self.task_type = pipeline_config.task_type
        self.model_size = pipeline_config.model_size
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.dist_config = dist_config
        self.weight_dtype = (
            model_config.weight_dtype
            if model_config.weight_dtype is not None
            else pipeline_config.default_config.get("param_dtype", DEFAULT_RUN_DTYPE)
        )
        if isinstance(self.weight_dtype, str):
            if self.weight_dtype == "bfloat16":
                self.weight_dtype = torch.bfloat16
            elif self.weight_dtype == "float16":
                self.weight_dtype = torch.float16
            elif self.weight_dtype == "default":
                self.weight_dtype = torch.float16

        if self.dist_config.ulysses_size > 1:
            assert (
                self.self.pipeline_config.default_config.num_heads % self.dist_config.ulysses_size == 0
            ), f"`{self.self.pipeline_config.default_config.num_heads=}` cannot be divided evenly by `{self.dist_config.ulysses_size=}`."
        if dist_config.use_sp:
            self.sp_size = dist_config.world_size
        else:
            self.sp_size = 1
        log.info(f"KsanaDiffusionModel init with {self.model_config}")

    @time_range
    def load(
        self,
        comfy_model_path: str = None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_operations=None,
        load_device=None,
        offload_device=None,
        checkpoint_dir=None,
        subfolder=None,
        lora_dir=None,
        shard_fn=None,
    ):
        assert self.model_name in ["wan2.2"], "only support wan2.2 yet"
        if "wan" in self.model_name:
            self.load_wan_model(
                comfy_model_path=comfy_model_path,
                comfy_model_config=comfy_model_config,
                comfy_model_state_dict=comfy_model_state_dict,
                comfy_operations=comfy_operations,
                load_device=load_device,
                offload_device=offload_device,
                checkpoint_dir=checkpoint_dir,
                subfolder=subfolder,
                lora_dir=lora_dir,
                shard_fn=None,
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
        comfy_operations=None,
        load_device=None,
        offload_device=None,
        checkpoint_dir=None,
        subfolder=None,
        lora_dir=None,
        shard_fn=None,
    ):
        # with ksanaProfiler("KsanaDiffusionModel.load_wan_model"):
        if comfy_model_config is not None and comfy_model_state_dict is not None:
            log.info(
                f"load from comfy_model_path:{comfy_model_path}, comfy_model_config:{comfy_model_config}, "
                f"comfy_operations:{comfy_operations}, load_device:{load_device}, offload_device:{offload_device}"
            )
            default_model_config = self.pipeline_config.default_config
            in_dim = comfy_model_config.get("in_dim", default_model_config.get("in_dim", 16))
            out_dim = comfy_model_config.get("out_dim", default_model_config.get("out_dim", 16))
            start = time.time()
            self.model = WanModel(
                model_type=self.task_type,
                patch_size=default_model_config.patch_size,
                text_len=default_model_config.text_len,
                in_dim=in_dim,
                dim=default_model_config.dim,
                ffn_dim=default_model_config.ffn_dim,
                freq_dim=default_model_config.freq_dim,
                out_dim=out_dim,
                num_heads=default_model_config.num_heads,
                num_layers=default_model_config.num_layers,
                window_size=default_model_config.window_size,
                qk_norm=default_model_config.qk_norm,
                cross_attn_norm=default_model_config.cross_attn_norm,
                eps=default_model_config.eps,
                comfy_operations=comfy_operations,
                device=offload_device,
                dtype=self.weight_dtype,
            )
            stop1 = time.time()
            load_result = self.model.load_state_dict(comfy_model_state_dict, strict=False)
            stop2 = time.time()
            log.warning(
                f"load_result: missing keys:{load_result.missing_keys}, unexpected keys:{load_result.unexpected_keys}"
            )
            log.info(f"create model takes: {(stop1 - start):.2f}, load states takes {(stop2 - stop1):.2f} seconds")

        else:
            log.info(f"load from checkpoint_dir:{checkpoint_dir}, subfolder:{subfolder}")
            self.model = WanModel.from_pretrained(checkpoint_dir, subfolder=subfolder)

        log.debug(f"model: {self.model}")
        self.apply_torch_compile(self.model_config.torch_compile_config)

        self.model.eval().requires_grad_(False)
        self.model = self.prepare_distributed_model(
            model=self.model, use_sp=self.dist_config.use_sp, dit_fsdp=self.dist_config.dit_fsdp, shard_fn=shard_fn
        )

        if lora_dir:
            log.info(f"load lora from {lora_dir}")
            self.model.set_keep_in_fp32_modules()
            self.model = load_and_merge_lora_weight_from_safetensors(self.model, lora_dir)
            model_safe_downcast(
                self.model,
                dtype=self.weight_dtype,
                keep_in_fp32_modules=[],
                keep_in_fp32_parameters=self.model._keep_in_fp32_params,
            )

    def prepare_distributed_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype=False):
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        # else:
        #     if convert_model_dtype:
        #         model.to(self.param_dtype)
        #     if not self.init_on_cpu:
        #         model.to(self.device)

        return model

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    @property
    def default_model_config(self):
        return self.pipeline_config.default_config

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_default_pipeline_config(model_name, task_type, model_size):
        if model_name == "wan2.2":
            config_type = f"{task_type}-{model_size}"
            return WAN2_2_CONFIGS[config_type]
        else:
            raise ValueError(f"model_name {model_name} not supported")

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
