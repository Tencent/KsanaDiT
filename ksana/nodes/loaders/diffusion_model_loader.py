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

import gc
import os

import torch

from ksana.config import KsanaLoraConfig, KsanaModelConfig
from ksana.memory import PinnedMemoryManager
from ksana.models import KsanaQwenImageModel, KsanaWanModel, KsanaWanVaceModel
from ksana.models.model_key import KsanaModelKey
from ksana.operations import build_ops
from ksana.settings import load_default_settings
from ksana.utils import is_file_or_dir, log
from ksana.utils.lora import load_state_dict_and_merge_lora
from ksana.utils.profile import time_range

from ..core.base_node import KsanaLoadNode
from ..core.node_factory import KsanaLoaderNodeFactory
from ..core.node_types import KsanaDispatchPolicy


@KsanaLoaderNodeFactory.register(
    [
        KsanaModelKey.Wan2_2_T2V_14B,
        KsanaModelKey.Wan2_2_I2V_14B,
        KsanaModelKey.Wan2_1_VACE_14B,
        KsanaModelKey.QwenImage_T2I,
        KsanaModelKey.QwenImage_Edit,
    ],
)
class DiffusionLoaderNode(KsanaLoadNode):
    """加载 Diffusion 模型。

    kwargs 由 Executor.run_loader_node() 自动注入 dist_config / shard_fn，
    Pipeline 只需传 model_path / model_config / lora_config。
    ComfyUI 适配层可额外传 model_patch_path 用于合并补丁权重（如 VACE）。
    """

    dispatch_policy = KsanaDispatchPolicy.ALL_ALL_ALL

    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.Wan2_2_I2V_14B: KsanaWanModel,
        KsanaModelKey.Wan2_2_T2V_14B: KsanaWanModel,
        KsanaModelKey.Wan2_1_VACE_14B: KsanaWanVaceModel,
        KsanaModelKey.QwenImage_T2I: KsanaQwenImageModel,
        KsanaModelKey.QwenImage_Edit: KsanaQwenImageModel,
    }

    _pinned_memory_manager: PinnedMemoryManager = None

    # ── 输入校验 ──────────────────────────────────────────────────────────

    @staticmethod
    def _valid_input_model_path(model_path: str | list[str]):
        load_model_path_or_files = model_path
        # two cases:
        # 1. [comfy load] model_path is a safetensors file or list of safetensors files
        # 2. [local load] model_path can be one dir or one file, or  a list of files, or a list of dirs
        if isinstance(model_path, (list, tuple)):
            if any(not is_file_or_dir(one) for one in model_path):
                raise ValueError(f"model_path must be list of files or dirs, but got model_path:{model_path}")
        elif is_file_or_dir(model_path):
            load_model_path_or_files = [model_path]
        else:
            raise ValueError(f"model_path must be a file/dir or a list of file/dir, but got {model_path}")
        return load_model_path_or_files

    @staticmethod
    def _valid_input_model_patch_path(model_patch_path: None | list[str], model_count: int) -> list[str | None] | None:
        if model_patch_path is None:
            return None
        if isinstance(model_patch_path, str):
            return [model_patch_path] + [None] * (model_count - 1)
        elif isinstance(model_patch_path, list):
            if len(model_patch_path) > model_count:
                raise ValueError(
                    f"len of model_patch_path list must not exceed {model_count}, but got {model_patch_path}"
                )
            model_patch_path = model_patch_path + [None] * (model_count - len(model_patch_path))
            if len(model_patch_path) == 0:
                return None
            for one_path in model_patch_path:
                if one_path is not None and not isinstance(one_path, str):
                    raise ValueError(f"model_patch_path[i] must be str, but got {one_path}")
        else:
            raise ValueError(f"model_patch_path must be list of str, but got {model_patch_path}")
        return model_patch_path

    @staticmethod
    def _valid_input_lora(
        model_key, lora_config: None | list[list[KsanaLoraConfig]] | list[KsanaLoraConfig], model_count: int
    ) -> list:
        if lora_config is None:
            return None
        if not isinstance(lora_config, list):
            raise ValueError(f"lora_config must be list of list of KsanaLoraConfig, but got {lora_config}")
        if len(lora_config) == 0:
            return None
        if all(isinstance(i, KsanaLoraConfig) for i in lora_config):
            lora_config = [lora_config]

        return_list = []
        if len(lora_config) != model_count:
            raise ValueError(f"len of lora_config list must be {model_count} for {model_key}, but got {lora_config}")
        for one_list in lora_config:
            if not isinstance(one_list, (list, tuple)) and not isinstance(one_list, KsanaLoraConfig):
                raise ValueError(
                    f"lora_config[i] must be list of KsanaLoraConfig or KsanaLoraConfig, but got {one_list}"
                )
            if isinstance(one_list, KsanaLoraConfig):
                one_list = [one_list]
            return_list.append(one_list)
        return return_list

    # ── 权重加载 ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_state_dict(
        model_key,
        default_settings,
        model_path: str,
        run_dtype,
        device,
        lora_config: None | list[KsanaLoraConfig] = None,
        model_patch_path: str = None,
    ):
        if model_key in [KsanaModelKey.QwenImage_T2I, KsanaModelKey.QwenImage_Edit] and os.path.isdir(model_path):
            if getattr(default_settings.diffusion, "transformer_subdir", None) is None:
                raise ValueError(
                    f"transformer_subdir must be set in diffusion section of default_settings for"
                    f" {model_key}, but got {default_settings.diffusion}"
                )
            transformer_dir = os.path.join(model_path, default_settings.diffusion.transformer_subdir)
            return load_state_dict_and_merge_lora(
                transformer_dir, lora_config, run_dtype, device=device, model_patch_path=model_patch_path
            )
        else:
            return load_state_dict_and_merge_lora(
                model_path, lora_config, run_dtype, device=device, model_patch_path=model_patch_path
            )

    # ── 主加载逻辑 ────────────────────────────────────────────────────────

    @time_range
    # TODO(test): 需要添加一个本地测试用例来验证 model_patch_path 的加载功能
    def run(self, model_key, *, model_pool, device_ctx, **kwargs):
        model_path = kwargs.pop("model_path")
        model_patch_path = kwargs.pop("model_patch_path", None)
        model_config: KsanaModelConfig = kwargs.pop("model_config", None)
        lora_config = kwargs.pop("lora_config", None)
        dist_config = kwargs.pop("dist_config", None)
        shard_fn = kwargs.pop("shard_fn", None)
        comfy_bar_callback = kwargs.pop("comfy_bar_callback", None)

        device = device_ctx.offload_device
        offload_device = device_ctx.offload_device

        log.info(f"{model_key} loading diffusion model from: {model_path}")
        load_model_path_or_files = self._valid_input_model_path(model_path)
        model_patch_path = self._valid_input_model_patch_path(model_patch_path, len(load_model_path_or_files))
        list_of_loras_list = self._valid_input_lora(model_key, lora_config, len(load_model_path_or_files))
        default_settings = load_default_settings(model_key, with_lora=list_of_loras_list is not None)
        device = device or torch.device("cuda")

        if DiffusionLoaderNode._pinned_memory_manager is None:
            DiffusionLoaderNode._pinned_memory_manager = PinnedMemoryManager()
            log.info("Initialized shared PinnedMemoryManager for DiffusionLoaderNode")

        res = []
        for i in range(len(load_model_path_or_files)):
            one_model_path = load_model_path_or_files[i]
            loras_list = list_of_loras_list[i] if list_of_loras_list is not None else None
            one_patch_path = model_patch_path[i] if model_patch_path is not None else None
            model_state_dict = self._load_state_dict(
                model_key, default_settings, one_model_path, model_config.run_dtype, device, loras_list, one_patch_path
            )
            model_class = self._MAP_KEY_TO_MODEL_CLASS.get(model_key)
            if model_class is None:
                raise ValueError(f"model_key {model_key} not supported")
            model = model_class(
                model_key,
                model_config,
                dist_config,
                default_settings,
                pinned_memory_manager=DiffusionLoaderNode._pinned_memory_manager,
            )
            model_state_dict = model.preprocess_model_state_dict(model_state_dict)
            # TODO(rock): get weight dtype from model_state_dict and judge linear_backend use fp8_gemm or not
            operations = build_ops(
                model_config.run_dtype,
                model_state_dict,
                attention_config=model_config.attention_config,
                linear_backend=model_config.linear_backend,
                rms_dtype=model_config.rms_dtype,
            )
            # qkv fusion is disabled when loading multiple diffusion models(like high + low) due to RAM OOM
            operations.disable_qkv_fusion = len(load_model_path_or_files) > 1
            log.info(f"loading {model_key} to device:{device}, offload_device:{offload_device}")

            model.load(
                model_state_dict=model_state_dict,
                operations=operations,
                load_device=device,
                offload_device=offload_device,
            )
            log.debug(f"{model_key} model: {model.model}")
            model.load_state_dict(model_state_dict, strict=False)
            model.enable_only_infer()
            model.prepare_distributed_model(shard_fn)
            model.apply_dynamic_fp8_quant(
                linear_backend=model_config.linear_backend,
                load_device=device,
                model_state_dict=model_state_dict,
            )
            # Free state_dict early to reduce peak memory when loading multiple models(high and low noise models)
            del model_state_dict
            gc.collect()

            model.apply_torch_compile(model_config.torch_compile_config)
            # Note: apply_pinned_memory must be called after apply_torch_compile
            model.apply_pinned_memory(offload_device)

            if offload_device is not None:
                model = model.to(offload_device)
            res.append(model)
            if comfy_bar_callback is not None:
                comfy_bar_callback()

        loaded_model = res[0] if len(res) == 1 else res
        model_pool.update_model_with_key(model_key, loaded_model)
