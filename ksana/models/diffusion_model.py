from abc import abstractmethod

import torch
import torch.distributed as dist

from ksana.operations import KsanaLinearBackend, build_ops

from ..config import KsanaDistributedConfig, KsanaModelConfig
from ..models.model_key import KsanaModelKey
from ..utils import log, time_range
from ..utils.load import load_state_dict
from ..utils.quantize import maybe_apply_dynamic_fp8_quant
from ..utils.torch_compile import apply_torch_compile
from .base_model import KsanaModel
from .wan import WanModel


class KsanaDiffusionModel(KsanaModel):
    def __init__(
        self,
        model_key: KsanaModelKey,
        model_config: KsanaModelConfig,
        dist_config: KsanaDistributedConfig | None,
        default_settings,
    ):
        super().__init__(model_key, default_settings)
        self.model_config = model_config
        self.dist_config = dist_config or KsanaDistributedConfig()
        sp_size = self.dist_config.ulysses_size
        num_heads = self.default_settings.diffusion.num_heads
        if self.dist_config.ulysses_size > 1:
            assert num_heads % sp_size == 0, f"`{num_heads}` can't be divided by `{sp_size}`."
        if self.dist_config.use_sp:
            self.sp_size = self.dist_config.num_gpus
        else:
            self.sp_size = 1
        log.info(f"KsanaDiffusionModel init with model_config:{self.model_config}")
        log.info(f"dist_config:{self.dist_config}")
        log.info(f"default_settings:{self.default_settings}")

        self._pinned_params = {}
        # NOTE: use more memory when using pinned memory.
        self._use_pinned_memory = True
        self._preallocated_pinned_memory = False

    @time_range
    def preallocate_pinned_memory(self, offload_device):
        if self._preallocated_pinned_memory or not self._use_pinned_memory or offload_device.type != "cpu":
            return

        # fp8_gemm_dynamic uses torchao Float8Tensor weights which are not compatible with
        # our pinned-memory swap (it mutates `.data` and can error with incompatible tensor type).
        if self.model_config.linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            return

        # 按dtype分组分配统一的 pinned memory buffer
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
            f"Unified pinned buffer allocated successfully, "
            f"total: {total_memory_gb:.2f} GB across {len(dtype_groups)} dtype(s)"
        )
        self._preallocated_pinned_memory = True

        # Promptly migrate model to pinned memory to avoid duplicate copies in memory
        self._migrate_model_to_pinned_memory()

    def _migrate_model_to_pinned_memory(self):
        cnt = 0

        def _process_tensor(name, tensor, key_prefix=""):
            nonlocal cnt
            if not tensor.is_pinned():
                key = f"{key_prefix}{name}" if key_prefix else name
                self._pinned_params[key].copy_(tensor)
                tensor.data = self._pinned_params[key]
                cnt += 1

        for name, param in self.model.named_parameters():
            _process_tensor(name, param)

        for name, buffer in self.model.named_buffers():
            _process_tensor(name, buffer, key_prefix=self.buffer_prefix)

        log.debug(f"{cnt} parameters migrated to pinned memory")

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

    def prepare_distributed_model(self, model, use_sp, dit_fsdp, shard_fn, convert_model_dtype=False):
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            model = shard_fn(model)
        return model

    def to(self, device=None, **kwargs):
        """
        将模型移动到指定设备
        支持 GPU 和 CPU (pinned memory) 之间的高效切换

        注意: 使用 pinned memory 时不支持 dtype 转换
        """
        if self._use_pinned_memory and "dtype" in kwargs:
            raise ValueError("使用 pinned memory 时不支持 dtype 转换。")

        if device is not None and self.model_config.linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
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
        if not self._use_pinned_memory or self.model_config.linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            self.model.to("cpu")
            return

        # 统一处理 parameters 和 buffers
        def _process_tensor(name, tensor, key_prefix=""):
            if tensor.is_cuda:
                key = f"{key_prefix}{name}" if key_prefix else name
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
        if not self._use_pinned_memory or self.model_config.linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            self.model.to(device)
            return

        # 统一处理 parameters 和 buffers
        def _process_tensor(name, tensor, key_prefix=""):
            if not tensor.is_cuda:
                if tensor.is_pinned():
                    # 从 pinned memory 直接传输到 GPU（快）
                    tensor.data = tensor.to(device, non_blocking=True)
                else:
                    raise RuntimeError(
                        f"Tensor {name} is not pinned. it should have been migrated to pinned memory in advance."
                    )

        for name, param in self.model.named_parameters():
            _process_tensor(name, param)

        for name, buffer in self.model.named_buffers():
            _process_tensor(name, buffer, key_prefix=self.buffer_prefix)

        torch.cuda.synchronize(device)

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

    # @staticmethod
    # def get_model_type(full_model_name: str):
    #     """
    #     Get the model type from the state dict.
    #     model_name: wan2.2, wan2.1
    #     model_type: t2v or s2v or i2v or ti2v
    #     model_size: A14B, 5B. etc.
    #     return (model_name, model_type, model_size)
    #     """
    #     lower_model_name = full_model_name.lower()
    #     if any_key_in_str(QWEN_IMAGE, lower_model_name) is not None:
    #         model_name = QWEN_IMAGE[0]
    #         model_type = "t2i"
    #         model_size = "20B"
    #         log.info(f"model_name:{model_name}, task_type: {model_type}, model_size: {model_size}")
    #         return model_name, model_type, model_size

    #     idx = any_key_in_str(X2V_TYPES, lower_model_name)
    #     model_name = None
    #     if idx is None:
    #         raise RuntimeError(f"can not detect model_type:{X2V_TYPES} from model_name:{full_model_name}")
    #     else:
    #         model_type = X2V_TYPES[idx]
    #     if any_key_in_str(WAN2_2, lower_model_name) is not None:
    #         model_name = WAN2_2[0]
    #     else:
    #         raise RuntimeError(f"can not detect model_name:{WAN2_2} from model_name:{full_model_name}")

    #     model_size = None
    #     if model_name == WAN2_2[0]:
    #         if "14b" in lower_model_name:
    #             model_size = "A14B"
    #         elif "5b" in lower_model_name:
    #             model_size = "5B"
    #     elif model_name == "wan2.1":
    #         if "14b" in lower_model_name:
    #             model_size = "14B"
    #         elif "1.3b" in lower_model_name or "1_3b" in lower_model_name:
    #             model_size = "1.3B"
    #     else:
    #         raise RuntimeError(f"can not detect model_size from model_name:{model_name}, model_name:{model_name}")

    #     log.info(f"model_name:{model_name}, task_type: {model_type}, model_size: {model_size}")
    #     return model_name, model_type, model_size


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
        # TODO(TJ): remove this flag
        self._is_high = True

    def set_as_high_or_low(self, is_high):
        self._is_high = is_high

    @time_range
    def load(
        self,
        model_state_dict: dict,
        load_device=None,
        offload_device=None,
        shard_fn=None,
    ):
        # TODO(rock): get weight dtype from model_state_dict and judge linear_backend use fp8_gemm or not
        operations = build_ops(
            self.run_dtype,
            model_state_dict,
            attention_config=self.model_config.attention_config,
            linear_backend=self.model_config.linear_backend,
            rms_dtype=self.model_config.rms_dtype,
        )
        log.info(f"load_device:{load_device}, offload_device:{offload_device}")

        default_diffusion_settings = self.default_settings.diffusion
        in_dim, out_dim = self._get_in_out_dim(model_state_dict)
        log.info(f"in_dim:{in_dim}, out_dim:{out_dim}")
        with time_range(f"create_model_{self.model_key}"):
            self.model = WanModel(
                is_i2v_type=self.model_key.is_i2v_type(),
                patch_size=default_diffusion_settings.patch_size,
                text_len=self.default_settings.text_encoder.text_len,
                in_dim=in_dim,
                dim=default_diffusion_settings.dim,
                ffn_dim=default_diffusion_settings.ffn_dim,
                freq_dim=default_diffusion_settings.freq_dim,
                out_dim=out_dim,
                num_heads=default_diffusion_settings.num_heads,
                num_layers=default_diffusion_settings.num_layers,
                window_size=default_diffusion_settings.window_size,
                qk_norm=default_diffusion_settings.qk_norm,
                cross_attn_norm=default_diffusion_settings.cross_attn_norm,
                eps=default_diffusion_settings.eps,
                operations=operations,
                device=offload_device,
                dtype=self.run_dtype,
                sp_size=self.dist_config.ulysses_size,
            )
        log.debug(f"model: {self.model}")
        load_state_dict(self.model, model_state_dict, strict=False)

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


class KsanaQwenImageModel(KsanaDiffusionModel):
    pass
    # @time_range
    # def load(
    #     self,
    #     model_state_dict: dict,
    #     *,
    #     load_device=None,
    #     offload_device=None,
    #     shard_fn=None,
    #     **kwargs,
    # ):
    #     load_device = load_device or torch.device("cuda")
    #     log.info(f"{self.model_config=}")

    #     operations = build_ops(
    #         self.run_dtype,
    #         model_state_dict,
    #         attention_config=self.model_config.attention_config,
    #         linear_backend=self.model_config.linear_backend,
    #         rms_dtype=self.model_config.rms_dtype,
    #     )

    #     self.model = QwenImageTransformer2DModel(
    #         patch_size=input_model_config.patch_size,
    #         in_channels=input_model_config.in_channels,
    #         out_channels=input_model_config.out_channels,
    #         num_layers=input_model_config.num_layers,
    #         attention_head_dim=input_model_config.attention_head_dim,
    #         num_attention_heads=input_model_config.num_attention_heads,
    #         joint_attention_dim=input_model_config.joint_attention_dim,
    #         axes_dims_rope=tuple(input_model_config.axes_dims_rope),
    #         operations=operations,
    #         device=offload_device,
    #         dtype=self.run_dtype,
    #         sp_size=self.dist_config.ulysses_size,
    #     )

    #     self.model.to(load_device, dtype=self.run_dtype)

    #     load_state_dict(self.model, model_state_dict, strict=False)

    #     self.model.eval().requires_grad_(False)
    #     if shard_fn is not None and self.dist_config is not None:
    #         self.model = self.prepare_distributed_model(
    #             model=self.model,
    #             use_sp=self.dist_config.use_sp,
    #             dit_fsdp=self.dist_config.dit_fsdp,
    #             shard_fn=shard_fn,
    #         )

    #     self.model = apply_torch_compile(self.model, self.model_config.torch_compile_config)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     t: torch.Tensor,
    #     context: torch.Tensor,
    #     context_mask: torch.Tensor = None,
    #     img_shapes: list = None,
    #     txt_seq_lens: list = None,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     timestep = t / 1000.0

    #     out = self.model(
    #         hidden_states=x,
    #         encoder_hidden_states=context,
    #         encoder_hidden_states_mask=context_mask,
    #         timestep=timestep,
    #         img_shapes=img_shapes,
    #         txt_seq_lens=txt_seq_lens,
    #         return_dict=False,
    #     )

    #     if isinstance(out, (tuple, list)):
    #         return out[0]
    #     return out
