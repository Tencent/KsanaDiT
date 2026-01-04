import types
from abc import abstractmethod

import torch
import torch.distributed as dist

from ..distributed import sp_attn_forward
from ..utils import log, time_range
from ..utils.quantize import maybe_apply_dynamic_fp8_quant
from ..utils.torch_compile import apply_torch_compile

from .wan import WanModel
from .wan.configs import WAN2_2_CONFIGS
from ..config import KsanaModelConfig, KsanaDistributedConfig
from ksana.operations import build_ops, KsanaAttentionBackend
from .base_model import KsanaModel
from ..utils.types import any_key_in_str
from ..models.model_key import KsanaModelKey, WAN2_2, WAN2_1, X2V_TYPES


class KsanaDiffusionModel(KsanaModel):
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

        self._pinned_params = {}
        # NOTE: use more memory when using pinned memory.
        self._use_pinned_memory = True

    @time_range
    def preallocate_pinned_memory(self, offload_device):
        # fp8_gemm_dynamic uses torchao Float8Tensor weights which are not compatible with
        # our pinned-memory swap (it mutates `.data` and can error with incompatible tensor type).
        if getattr(self.model_config, "linear_backend", None) == "fp8_gemm_dynamic":
            return

        # 按dtype分组分配统一的 pinned memory buffer
        if not self._use_pinned_memory or offload_device.type != "cpu":
            return

        dtype_groups = {}  # {dtype: [(name, shape, numel), ...]}

        for name, param in self.model.named_parameters():
            dtype = param.dtype
            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append((name, param.shape, param.numel()))

        self.buffer_prefix = "buffer_"
        for name, buffer in self.model.named_buffers():
            buffer_key = f"{self.buffer_prefix}{name}"
            dtype = buffer.dtype
            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append((buffer_key, buffer.shape, buffer.numel()))

        # 为每个dtype分配独立的unified buffer
        self._unified_pinned_buffers = {}  # 保存所有buffer引用,防止被GC
        total_memory_gb = 0

        for dtype, params in dtype_groups.items():
            total_elements = sum(numel for _, _, numel in params)
            memory_gb = total_elements * torch.tensor([], dtype=dtype).element_size() / 1024**3
            total_memory_gb += memory_gb

            log.info(f"Allocating {dtype} pinned buffer: {total_elements} elements ({memory_gb:.2f} GB)")

            # 为当前dtype分配unified buffer
            unified_buffer = torch.empty(total_elements, dtype=dtype, pin_memory=True)
            self._unified_pinned_buffers[dtype] = unified_buffer

            # 切片分配给各个参数
            offset = 0
            for name, shape, numel in params:
                self._pinned_params[name] = unified_buffer[offset : offset + numel].view(shape)
                offset += numel

        log.info(
            f"Unified pinned buffer allocated successfully, total: {total_memory_gb:.2f} GB across {len(dtype_groups)} dtype(s)"
        )

    @abstractmethod
    def get_model_key(self) -> KsanaModelKey:
        pass

    @abstractmethod
    @time_range
    def load(
        self,
        model_state_dict: dict,
        *,
        model_config: KsanaModelConfig = None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        pass

    @time_range
    def do_apply_torch_compile(self, torch_compile_config=None):
        """Apply torch compile to the model using the standalone function."""
        self.model = apply_torch_compile(self.model, torch_compile_config)

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

    def load_warm_up(self, device, offload_device):
        self.preallocate_pinned_memory(offload_device)
        self.to(device=device)
        self.to(device=offload_device)

    def to(self, device=None, **kwargs):
        """
        将模型移动到指定设备
        支持 GPU 和 CPU (pinned memory) 之间的高效切换

        注意: 使用 pinned memory 时不支持 dtype 转换
        """
        if self._use_pinned_memory and "dtype" in kwargs:
            raise ValueError("使用 pinned memory 时不支持 dtype 转换。")

        if device is not None and getattr(self.model_config, "linear_backend", None) == "fp8_gemm_dynamic":
            self.model.to(device, **kwargs)
            return self

        if not self._use_pinned_memory or device is None:
            self.model.to(device, **kwargs)
            return self

        if isinstance(device, str):
            device = torch.device(device)
        current_device = self.device

        # GPU -> CPU: 保存到 pinned memory
        if current_device.type == "cuda" and device.type == "cpu":
            self._offload_to_pinned_memory()
            if kwargs:
                self.model.to(**kwargs)
        # CPU -> GPU: 从 pinned memory 加载
        elif current_device.type == "cpu" and device.type == "cuda":
            self._load_from_pinned_memory(device)
            if kwargs:
                self.model.to(**kwargs)
        else:
            self.model.to(device, **kwargs)

        return self

    def _offload_to_pinned_memory(self):
        """将模型参数从 GPU offload 到 CPU 的 pinned memory"""
        if not self._use_pinned_memory or getattr(self.model_config, "linear_backend", None) == "fp8_gemm_dynamic":
            self.model.to("cpu")
            return

        # 统一处理 parameters 和 buffers
        def _process_tensor(name, tensor, key_prefix=""):
            if tensor.is_cuda:
                key = f"{key_prefix}{name}" if key_prefix else name
                # 创建或复用 pinned memory
                if key not in self._pinned_params:
                    self._pinned_params[key] = torch.empty_like(tensor, pin_memory=True)
                # GPU -> Pinned Memory
                self._pinned_params[key].copy_(tensor, non_blocking=True)
                # 直接使用 pinned memory
                tensor.data = self._pinned_params[key]

        for name, param in self.model.named_parameters():
            _process_tensor(name, param)

        for name, buffer in self.model.named_buffers():
            _process_tensor(name, buffer, key_prefix=self.buffer_prefix)

        torch.cuda.synchronize()

    def _load_from_pinned_memory(self, device: torch.device):
        """从 CPU 的 pinned memory 加载模型参数到 GPU"""
        if not self._use_pinned_memory or getattr(self.model_config, "linear_backend", None) == "fp8_gemm_dynamic":
            self.model.to(device)
            return

        # 统一处理 parameters 和 buffers
        def _process_tensor(name, tensor, key_prefix=""):
            if not tensor.is_cuda:
                if tensor.is_pinned():
                    # 从 pinned memory 直接传输到 GPU（快）
                    tensor.data = tensor.to(device, non_blocking=True)
                else:
                    # 如果不在 pinned memory，先转到 pinned memory
                    key = f"{key_prefix}{name}" if key_prefix else name
                    if key not in self._pinned_params:
                        self._pinned_params[key] = torch.empty_like(tensor, pin_memory=True)
                    self._pinned_params[key].copy_(tensor)
                    tensor.data = self._pinned_params[key].to(device, non_blocking=True)

        for name, param in self.model.named_parameters():
            _process_tensor(name, param)

        for name, buffer in self.model.named_buffers():
            _process_tensor(name, buffer, key_prefix=self.buffer_prefix)

        torch.cuda.synchronize(device)

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
        if model_name in WAN2_2:
            config_type = f"{task_type}-{model_size}"
            return WAN2_2_CONFIGS[config_type]
        else:
            raise ValueError(f"model_name {model_name} not supported")

    @property
    def full_model_name(self):
        return f"{self.model_name}_{self.task_type}_{self.model_size}"

    @staticmethod
    def get_model_type(full_model_name: str):
        """
        Get the model type from the state dict.
        model_name: wan2.2, wan2.1
        model_type: t2v or s2v or i2v or ti2v
        model_size: A14B, 5B. etc.
        return (model_name, model_type, model_size)
        """
        lower_model_name = full_model_name.lower()
        idx = any_key_in_str(X2V_TYPES, lower_model_name)
        model_name = None
        if idx is None:
            raise RuntimeError(f"can not detect model_type:{X2V_TYPES} from model_name:{full_model_name}")
        else:
            model_type = X2V_TYPES[idx]
        if any_key_in_str(WAN2_2, lower_model_name) is not None:
            model_name = WAN2_2[0]
        else:
            raise RuntimeError(f"can not detect model_name:{WAN2_2} from model_name:{full_model_name}")

        model_size = None
        if model_name == WAN2_2[0]:
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
            raise RuntimeError(f"can not detect model_size from model_name:{model_name}, model_name:{model_name}")

        log.info(f"model_name:{model_name}, task_type: {model_type}, model_size: {model_size}")
        return model_name, model_type, model_size


class KsanaWanModel(KsanaDiffusionModel):
    """
    Wan model class for Ksana diffusion models.
    """

    def _get_in_out_dim(self, state_dict, key_prefix=""):
        in_dim = state_dict["{}patch_embedding.weight".format(key_prefix)].shape[1]
        out_dim = state_dict["{}head.head.weight".format(key_prefix)].shape[0] // 4
        return in_dim, out_dim

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_high = True

    def set_as_high_or_low(self, is_high):
        self._is_high = is_high

    def get_model_key(self) -> KsanaModelKey:
        if self.model_name in WAN2_2:
            if self.task_type == "t2v":
                if self.model_size == "A14B":
                    return KsanaModelKey.Wan2_2_T2V_14B_HIGH if self._is_high else KsanaModelKey.Wan2_2_T2V_14B_LOW
                else:
                    raise RuntimeError(f"model_size {self.model_size} is not in {self.model_name} {self.task_type} yet")
            elif self.task_type == "i2v":
                if self.model_size == "A14B":
                    return KsanaModelKey.Wan2_2_I2V_14B_HIGH if self._is_high else KsanaModelKey.Wan2_2_I2V_14B_LOW
                else:
                    raise RuntimeError(f"model_size {self.model_size} is not in {self.model_name} {self.task_type} yet")
            else:
                raise ValueError(f"task_type {self.task_type} is not in supported list {X2V_TYPES} yet")
        elif self.model_name in WAN2_1:
            raise ValueError(f"model_name {self.model_name} not supported")
        else:
            raise ValueError(f"model_name {self.model_name} not supported")

    def load(
        self,
        model_state_dict: dict,
        load_device=None,
        offload_device=None,
        shard_fn=None,
    ):
        # TODO(rock): get weight dtype from model_state_dict and judge linear_backend is fp8_gemm or not
        # scaled_fp8_dtype = model_config.scaled_fp8
        # if (
        #     ksana_model_config.linear_backend == "default"
        #     and scaled_fp8_dtype is not None
        #     and "float8" in str(scaled_fp8_dtype)
        # ):
        #     print("linear_backend will use fp8_gemm")
        #     ksana_model_config.linear_backend = "fp8_gemm"
        # weight_dtype = next(iter(model_state_dict.values())).dtype
        # if weight_dtype != fp8
        #     log.warning(f"weight_dtype {weight_dtype} is not fp8, will use fp16_gemm linear_backend")
        #     self.model_config.linear_backend = "fp16_gemm"

        operations = build_ops(
            self.run_dtype,
            model_state_dict,
            attn_backend=KsanaAttentionBackend(self.model_config.attn_backend),
            linear_backend=self.model_config.linear_backend,
        )
        log.info(f"load_device:{load_device}, offload_device:{offload_device}")
        default_model_config = self.pipeline_config.default_config
        in_dim, out_dim = self._get_in_out_dim(model_state_dict)
        log.info(f"in_dim:{in_dim}, out_dim:{out_dim}")
        with time_range(f"create_model_{self.full_model_name}"):
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
        with time_range("load_state_dict"):
            load_result = self.model.load_state_dict(model_state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            log.warning(
                f"load_result: missing keys:{load_result.missing_keys}, unexpected keys:{load_result.unexpected_keys}"
            )

        self.model.eval().requires_grad_(False)
        self.model = self.prepare_distributed_model(
            model=self.model, use_sp=self.dist_config.use_sp, dit_fsdp=self.dist_config.dit_fsdp, shard_fn=shard_fn
        )

        maybe_apply_dynamic_fp8_quant(
            self.model,
            linear_backend=self.model_config.linear_backend,
            load_device=load_device,
        )

        # Note: apply torch compile should be after all weight loading and merging
        self.model = apply_torch_compile(self.model, self.model_config.torch_compile_config)
