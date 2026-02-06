# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from abc import abstractmethod

import torch

from ksana.operations.fuse_qkv import remap_state_dict_for_model

from ..accelerator import platform
from ..config import KsanaDistributedConfig, KsanaLinearBackend, KsanaModelConfig
from ..models.model_key import KsanaModelKey
from ..utils import log, time_range
from ..utils.load import load_state_dict, replace_key_in_state_dict
from ..utils.quantize import apply_dynamic_fp8_quant, find_fp8_info_from_state_dict
from ..utils.torch_compile import apply_torch_compile
from .base_model import KsanaModel
from .qwen import QwenImageTransformer2DModel
from .wan import VaceWanModel, WanModel

if platform.is_npu():
    import torch_npu  # pylint: disable=unused-import # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # pylint: disable=unused-import # noqa: F401


class KsanaDiffusionModel(KsanaModel):
    def __init__(
        self,
        model_key: KsanaModelKey,
        model_config: KsanaModelConfig,
        dist_config: KsanaDistributedConfig | None,
        default_settings,
        pinned_memory_manager=None,
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
        self._use_pinned_memory = platform.is_gpu()
        self._applied_pinned_memory = False
        self._allocated_blocks = []  # 存储从 manager 分配的内存块
        self._pinned_memory_manager = pinned_memory_manager

    def preprocess_model_state_dict(self, model_state_dict):
        return model_state_dict

    def _collect_dtype_groups(self):
        """收集模型中所有参数和 buffer，按 dtype 分组"""
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

        return dtype_groups

    def _ensure_block_available(self, blocks, current_block_idx):
        """确保有可用的 block，如果不够则申请新的"""
        if current_block_idx >= len(blocks):
            log.debug("Need additional block, requesting 1 more block")
            new_blocks = self._pinned_memory_manager.allocate_blocks(
                total_size_bytes=self._pinned_memory_manager.default_block_size_bytes
            )
            blocks.extend(new_blocks)
            self._allocated_blocks.extend(new_blocks)

    def _allocate_tensors_to_blocks(self, params, dtype):
        """
        将 tensor 分配到 blocks 中，确保每个 tensor 完整地在一个 block 中

        Args:
            params: [(name, shape, numel), ...] 参数列表
            dtype: 数据类型

        Returns:
            params的总大小（GB）
        """
        total_elements = sum(numel for _, _, numel in params)
        element_size = torch.tensor([], dtype=dtype).element_size()
        memory_bytes = total_elements * element_size
        memory_gb = memory_bytes / 1024**3

        log.info(f"Allocating {dtype} pinned buffer: {total_elements} elements ({memory_gb:.2f} GB)")

        # 从 manager 分配初始内存块（不再传递 dtype）
        blocks = self._pinned_memory_manager.allocate_blocks(total_size_bytes=memory_bytes)
        self._allocated_blocks.extend(blocks)

        # 确保每个 tensor 完整地放在一个 block 中
        current_block_idx = 0
        current_block_offset = 0  # 以字节为单位的偏移量

        for name, shape, numel in params:
            # 确保有可用的 block
            self._ensure_block_available(blocks, current_block_idx)

            current_block = blocks[current_block_idx]
            tensor_size_bytes = numel * element_size
            available_in_current_block = current_block.size_bytes - current_block_offset

            # 如果当前 block 剩余空间不足以放下整个 tensor，跳到下一个 block
            if available_in_current_block < tensor_size_bytes:
                log.debug(
                    f"Tensor {name} ({tensor_size_bytes} bytes) doesn't fit in current block "
                    f"(available: {available_in_current_block} bytes), moving to next block"
                )
                current_block_idx += 1
                current_block_offset = 0

                # 确保新的 block 可用
                self._ensure_block_available(blocks, current_block_idx)
                current_block = blocks[current_block_idx]

            # 从当前 block 分配 tensor（使用字节偏移，然后转换为目标 dtype）
            # 将 uint8 buffer 的一部分 view 为目标 dtype
            byte_view = current_block.buffer[current_block_offset : current_block_offset + tensor_size_bytes]
            tensor_view = byte_view.view(dtype).view(shape)
            self._pinned_params[name] = tensor_view

            # 更新偏移量（字节）
            current_block_offset += tensor_size_bytes

            # 如果当前 block 已满，移到下一个
            if current_block_offset >= current_block.size_bytes:
                current_block_idx += 1
                current_block_offset = 0

        log.info(
            f"Allocated {len(blocks)} blocks for {dtype}, "
            f"used {current_block_idx + (1 if current_block_offset > 0 else 0)} blocks"
        )

        return memory_gb

    @time_range
    def apply_pinned_memory(self, offload_device):
        """预分配 pinned memory，使用 PinnedMemoryManager 管理内存块"""
        if self._applied_pinned_memory or not self._use_pinned_memory or offload_device.type != "cpu":
            return

        # 如果没有提供 pinned_memory_manager，则跳过
        if self._use_pinned_memory and self._pinned_memory_manager is None:
            raise RuntimeError("Pinned memory manager is not provided")

        # fp8_gemm_dynamic uses torchao Float8Tensor weights which are not compatible with
        # our pinned-memory swap (it mutates `.data` and can error with incompatible tensor type).
        if self.model_config.linear_backend == KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            return

        # 收集所有参数和 buffer，按 dtype 分组
        dtype_groups = self._collect_dtype_groups()

        # 为每个 dtype 分配内存块
        total_memory_gb = 0
        for dtype, params in dtype_groups.items():
            memory_gb = self._allocate_tensors_to_blocks(params, dtype)
            total_memory_gb += memory_gb

        log.info(
            f"Pinned memory allocated from manager successfully, "
            f"total: {total_memory_gb:.2f} GB across {len(dtype_groups)} dtype(s), "
            f"using {len(self._allocated_blocks)} blocks"
        )
        self._applied_pinned_memory = True

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

    def __del__(self):
        """析构函数：释放分配的内存块"""
        self._release_pinned_memory()

    def _release_pinned_memory(self):
        """释放从 manager 分配的内存块"""
        if self._allocated_blocks:
            log.info(f"Releasing {len(self._allocated_blocks)} pinned memory blocks")
            self._pinned_memory_manager.release_blocks(self._allocated_blocks)
            self._allocated_blocks.clear()

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

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @time_range
    def load_state_dict(self, model_state_dict, strict=False):
        model_state_dict = remap_state_dict_for_model(self.model, model_state_dict, self.model_key.name)
        load_state_dict(self.model, model_state_dict, strict=strict)

    def enable_only_infer(self):
        if self.model is None:
            log.warning("model has not loaded yet!")
            return
        self.model.eval().requires_grad_(False)

    def prepare_distributed_model(self, shard_fn):
        if shard_fn is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # use_sp = self.dist_config.use_sp
        if self.dist_config.dit_fsdp:
            self.model = shard_fn(self.model)

    def apply_dynamic_fp8_quant(self, linear_backend, load_device, model_state_dict=None):
        if linear_backend != KsanaLinearBackend.FP8_GEMM_DYNAMIC:
            return
        # Check if weights are already in FP8 format (fp8_e4m3, fp8_e5m2, or scaled_fp8)
        weight_fp8_dtype, is_scaled_fp8 = find_fp8_info_from_state_dict(model_state_dict)
        if weight_fp8_dtype is not None or is_scaled_fp8:
            log.info(
                f"Skipping dynamic FP8 quantization: weights are already FP8 "
                f"(weight_fp8_dtype={weight_fp8_dtype}, is_scaled_fp8={is_scaled_fp8})"
            )
            return
        apply_dynamic_fp8_quant(self.model, load_device=load_device)

    @time_range
    def apply_torch_compile(self, torch_compile_config=None):
        """Apply torch compile to the model using the standalone function."""
        self.model = apply_torch_compile(self.model, torch_compile_config)

    @abstractmethod
    @time_range
    def load(
        self,
        *,
        model_state_dict: dict,
        operations,
        load_device=None,
        offload_device=None,
    ):
        pass

    @abstractmethod
    def run_dtype(self):
        raise NotImplementedError("run_dtype should be implemented by subclass.")

    @abstractmethod
    def dtype(self):
        raise NotImplementedError("dtype should be implemented by subclass.")

    @abstractmethod
    def device(self):
        raise NotImplementedError("device should be implemented by subclass.")


class KsanaWanModel(KsanaDiffusionModel):
    def _get_in_out_dim(self, state_dict, patch_size, key_prefix=""):
        patch_emb_weight_shape = state_dict["{}patch_embedding.weight".format(key_prefix)].shape
        if len(patch_emb_weight_shape) == 5:  # Conv3D weight for normal WanModel
            in_dim = patch_emb_weight_shape[1]
        elif len(patch_emb_weight_shape) == 2:  # Linear weight for TurboDiffusionWanModel
            in_dim = patch_emb_weight_shape[1] // math.prod(patch_size)
        else:
            raise ValueError(f"Unsupported patch_emb_weight_shape: {patch_emb_weight_shape}")
        out_dim = state_dict["{}head.head.weight".format(key_prefix)].shape[0] // 4
        return in_dim, out_dim

    def _is_turbo_diffusion_wan_model(self, state_dict, key_prefix=""):
        return (
            state_dict["{}patch_embedding.weight".format(key_prefix)].ndim == 2
        )  # Linear weight for TurboDiffusionWanModel

    def preprocess_model_state_dict(self, model_state_dict):
        # for turbo diffusion wan model
        old_pattern = ".self_attn.attn_op.local_attn.proj_l."
        new_pattern = ".self_attn.proj_l."
        return replace_key_in_state_dict(model_state_dict, old_pattern, new_pattern)

    @property
    def run_dtype(self):
        return self.model_config.run_dtype

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @time_range
    def load(
        self,
        *,
        model_state_dict: dict,
        operations,
        load_device=None,
        offload_device=None,
    ):
        default_diffusion_settings = self.default_settings.diffusion
        patch_size = default_diffusion_settings.patch_size
        in_dim, out_dim = self._get_in_out_dim(model_state_dict, patch_size)
        log.info(f"in_dim:{in_dim}, out_dim:{out_dim}")
        self.model = WanModel(
            is_i2v_type=self.model_key.is_i2v_type(),
            patch_size=patch_size,
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
            is_turbo_diffusion_wan_model=self._is_turbo_diffusion_wan_model(model_state_dict),
        )


class KsanaWanVaceModel(KsanaDiffusionModel):
    @property
    def run_dtype(self):
        return self.model_config.run_dtype

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def _get_in_out_dim(self, state_dict, key_prefix=""):
        in_dim = state_dict["{}patch_embedding.weight".format(key_prefix)].shape[1]
        out_dim = state_dict["{}head.head.weight".format(key_prefix)].shape[0] // 4
        return in_dim, out_dim

    def _detect_vace_params(self, model_state_dict):
        detected_vace_in_dim = None
        detected_vace_layers = None

        if "vace_patch_embedding.weight" in model_state_dict:
            detected_vace_in_dim = model_state_dict["vace_patch_embedding.weight"].shape[1]
            # Count vace_blocks from state dict
            vace_block_keys = [k for k in model_state_dict.keys() if k.startswith("vace_blocks.")]
            if vace_block_keys:
                # Extract block indices (vace_blocks.0.xxx, vace_blocks.1.xxx, etc.)
                block_indices = set()
                for k in vace_block_keys:
                    parts = k.split(".")
                    if len(parts) >= 2 and parts[1].isdigit():
                        block_indices.add(int(parts[1]))
                if block_indices:
                    detected_vace_layers = max(block_indices) + 1
            log.info(
                f"[VACE] Auto-detected from model weights: "
                f"vace_in_dim={detected_vace_in_dim}, vace_layers={detected_vace_layers}"
            )

        return detected_vace_in_dim, detected_vace_layers

    @time_range
    def load(
        self,
        *,
        model_state_dict: dict,
        operations,
        load_device=None,
        offload_device=None,
    ):
        default_diffusion_settings = self.default_settings.diffusion
        in_dim, out_dim = self._get_in_out_dim(model_state_dict)
        log.info(f"in_dim:{in_dim}, out_dim:{out_dim}")

        # Auto-detect VACE parameters from model weights
        detected_vace_in_dim, detected_vace_layers = self._detect_vace_params(model_state_dict)

        # Priority: detected from model > default_settings
        vace_in_dim = detected_vace_in_dim or getattr(default_diffusion_settings, "vace_in_dim", None)
        vace_layers = detected_vace_layers or getattr(default_diffusion_settings, "vace_layers", None)
        log.info(f"[VACE] Final config: vace_in_dim={vace_in_dim}, vace_layers={vace_layers}")

        start = time.time()
        self.model = VaceWanModel(
            model_type="vace",
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
            vace_layers=vace_layers,
            vace_in_dim=vace_in_dim,
            operations=operations,
            device=offload_device,
            dtype=self.run_dtype,
            sp_size=self.dist_config.ulysses_size,
        )
        log.debug(f"VACE model: {self.model}")
        log.info(f"create VACE model takes: {(time.time() - start):.2f} seconds")


class KsanaQwenImageModel(KsanaDiffusionModel):
    @time_range
    def load(
        self,
        *,
        model_state_dict: dict,
        operations,
        load_device=None,
        offload_device=None,
    ):
        default_diffusion_settings = self.default_settings.diffusion
        self.model = QwenImageTransformer2DModel(
            patch_size=default_diffusion_settings.patch_size,
            in_channels=default_diffusion_settings.in_channels,
            out_channels=default_diffusion_settings.out_channels,
            num_layers=default_diffusion_settings.num_layers,
            attention_head_dim=default_diffusion_settings.attention_head_dim,
            num_attention_heads=default_diffusion_settings.num_attention_heads,
            joint_attention_dim=default_diffusion_settings.joint_attention_dim,
            axes_dims_rope=tuple(default_diffusion_settings.axes_dims_rope),
            operations=operations,
            device=offload_device,
            dtype=self.run_dtype,
            sp_size=self.dist_config.ulysses_size,
        )

    @property
    def run_dtype(self):
        return self.model_config.run_dtype

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
        img_shapes: list = None,
        txt_seq_lens: list = None,
        **kwargs,
    ) -> torch.Tensor:
        timestep = t / 1000.0

        out = self.model(
            hidden_states=x,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )

        if isinstance(out, (tuple, list)):
            return out[0]
        return out
