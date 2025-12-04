import time
import types
from abc import ABC, abstractmethod
import os

import torch
import torch.distributed as dist

from ..distributed import sp_attn_forward
from ..utils import (  # , ksanaProfiler
    load_and_merge_lora_weight_from_safetensors,
    log,
    model_safe_downcast,
    time_range,
    load_sharded_safetensors,
    is_dir,
)

from ..utils.load import load_torch_file
from .wan import WanModel
from .wan.configs import WAN2_2_CONFIGS
from ..config import KsanaModelConfig, KsanaDistributedConfig
from ksana.operations import build_ops, AttentionBackendEnum


class KsanaDiffusionModel(ABC):
    def __init__(self, model_config: KsanaModelConfig, pipeline_config, dist_config: KsanaDistributedConfig):
        self.model_name = pipeline_config.model_name
        self.task_type = pipeline_config.task_type
        self.model_size = pipeline_config.model_size
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.dist_config = dist_config

        if self.dist_config.ulysses_size > 1:
            assert (
                self.pipeline_config.default_config.num_heads % self.dist_config.ulysses_size == 0
            ), f"`{self.pipeline_config.default_config.num_heads}` cannot be divided evenly by `{self.dist_config.ulysses_size}`."
        if dist_config.use_sp:
            self.sp_size = dist_config.num_gpus
        else:
            self.sp_size = 1
        log.info(f"KsanaDiffusionModel init with {self.model_config} {self.dist_config}")

    @abstractmethod
    @time_range
    def load(
        self,
        model_path,
        *,
        lora_file=None,
        model_config: KsanaModelConfig = None,
        comfy_model_config=None,
        comfy_model_state_dict=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        pass

    @time_range
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

    def comfy_load_model_weights(self, model, sd, unet_prefix=""):
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

    def prepare_distributed_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype=False):
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)

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
    def run_dtype(self):
        return self.model_config.run_dtype

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

        log.info(f"model_name:{model_name}, task_type: {model_type}, model_size: {model_size}")
        return model_name, model_type, model_size


class KsanaWanModel(KsanaDiffusionModel):
    """
    Wan model class for Ksana diffusion models.
    """

    def load(
        self,
        model_path: str,
        lora_file=None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        load_device=None,
        offload_device=None,
        shard_fn=None,
    ):
        if not (os.path.isfile(model_path) or is_dir(model_path)):
            raise ValueError(f"model_path {model_path} must be a file or dir")
        if comfy_model_state_dict is None:
            comfy_model_state_dict = (
                load_sharded_safetensors(f"{model_path}")
                if is_dir(model_path)
                else load_torch_file(model_path, device=load_device)
            )

        fp8_gemm, scaled_fp8 = (
            (True, torch.float8_e4m3fn) if self.model_config.linear_backend == "fp8_gemm" else (False, None)
        )

        operations = build_ops(
            self.run_dtype,
            backend=AttentionBackendEnum.from_string(self.model_config.attn_backend),
            fp8_gemm=fp8_gemm,
            scaled_fp8=scaled_fp8,
        )
        if comfy_model_config is None:
            comfy_model_config = {}
        log.info(
            f"load from model_path:{model_path}, lora_file:{lora_file}, comfy_model_config:{comfy_model_config}, "
            f"operations:{operations}, load_device:{load_device}, offload_device:{offload_device}"
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
            operations=operations,
            device=offload_device,
            dtype=self.run_dtype,
            sp_size=self.dist_config.ulysses_size,
        )
        log.debug(f"model: {self.model}")
        # TODO: use with time_range
        stop1 = time.time()
        load_result = self.model.load_state_dict(comfy_model_state_dict, strict=False)
        stop2 = time.time()
        if load_result.missing_keys or load_result.unexpected_keys:
            log.warning(
                f"load_result: missing keys:{load_result.missing_keys}, unexpected keys:{load_result.unexpected_keys}"
            )
        log.info(f"create model takes: {(stop1 - start):.2f}, load states takes {(stop2 - stop1):.2f} seconds")

        self.model.eval().requires_grad_(False)
        self.model = self.prepare_distributed_model(
            model=self.model, use_sp=self.dist_config.use_sp, dit_fsdp=self.dist_config.dit_fsdp, shard_fn=shard_fn
        )

        if lora_file:
            log.info(f"load lora from {lora_file}")
            self.model.set_keep_in_fp32_modules()
            self.model = load_and_merge_lora_weight_from_safetensors(self.model, lora_file)
            model_safe_downcast(
                self.model,
                dtype=self.run_dtype,
                keep_in_fp32_modules=[],
                keep_in_fp32_parameters=self.model._keep_in_fp32_params,
            )
        # Note: apply torch compile should be after all weight loading and merging
        self.apply_torch_compile(self.model_config.torch_compile_config)
