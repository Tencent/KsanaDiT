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

import os

import torch

from ..accelerator import platform
from ..config import KsanaLoraConfig, KsanaModelConfig
from ..memory import PinnedMemoryManager
from ..models import KsanaModel, KsanaQwenImageModel, KsanaWanModel, KsanaWanVaceModel
from ..models.model_key import KsanaModelKey
from ..operations import build_ops
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import is_file_or_dir, log
from ..utils.lora import load_state_dict_and_merge_lora
from ..utils.profile import time_range


@KsanaUnitFactory.register(
    KsanaUnitType.LOADER,
    [
        KsanaModelKey.Wan2_2_I2V_14B,
        KsanaModelKey.Wan2_2_T2V_14B,
        KsanaModelKey.Wan2_1_VACE_14B,
        KsanaModelKey.QwenImage_T2I,
    ],
)
class KsanaDiffusionLoaderUnit(KsanaLoaderUnit):
    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.Wan2_2_I2V_14B: KsanaWanModel,
        KsanaModelKey.Wan2_2_T2V_14B: KsanaWanModel,
        KsanaModelKey.Wan2_1_VACE_14B: KsanaWanVaceModel,
        KsanaModelKey.QwenImage_T2I: KsanaQwenImageModel,
    }

    _pinned_memory_manager: PinnedMemoryManager = None

    def _valid_input_model_path(self, model_path: str | list[str]):
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

    def _valid_input_vace_model(self, vace_model: None | list[str], model_count: int) -> list[str | None] | None:
        if vace_model is None:
            return None
        if isinstance(vace_model, str):
            return [vace_model] + [None] * (model_count - 1)
        elif isinstance(vace_model, list):
            if len(vace_model) > model_count:
                raise ValueError(f"len of vace_model list must not exceed {model_count}, but got {vace_model}")
            vace_model = vace_model + [None] * (model_count - len(vace_model))
            if len(vace_model) == 0:
                return None
            for one_model in vace_model:
                if one_model is not None and not isinstance(one_model, str):
                    raise ValueError(f"vace_model[i] must be str, but got {one_model}")
        else:
            raise ValueError(f"vace_model must be list of str, but got {vace_model}")
        return vace_model

    def _valid_input_lora(
        self, lora_config: None | list[list[KsanaLoraConfig]] | list[KsanaLoraConfig], model_count: int
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
            raise ValueError(
                f"len of lora_config list must be {model_count} for {self.model_key}, but got {lora_config}"
            )
        for one_list in lora_config:
            if not isinstance(one_list, (list, tuple)) and not isinstance(one_list, KsanaLoraConfig):
                raise ValueError(
                    f"lora_config[i] must be list of KsanaLoraConfig or KsanaLoraConfig, but got {one_list}"
                )
            if isinstance(one_list, KsanaLoraConfig):
                one_list = [one_list]
            return_list.append(one_list)
        return return_list

    def _load_state_dict(
        self,
        model_path: str,
        run_dtype,
        device,
        lora_config: None | list[KsanaLoraConfig] = None,
        vace_model: str = None,
    ):
        if self.model_key == KsanaModelKey.QwenImage_T2I and os.path.isdir(model_path):
            if getattr(self.default_settings.diffusion, "transformer_subdir", None) is None:
                raise ValueError(
                    f"transformer_subdir must be set in diffusion section of default_settings for"
                    f" {self.model_key}, but got {self.default_settings.diffusion}"
                )
            transformer_dir = os.path.join(model_path, self.default_settings.diffusion.transformer_subdir)
            return load_state_dict_and_merge_lora(transformer_dir, device=device, vace_model=vace_model)
        else:
            return load_state_dict_and_merge_lora(
                model_path, lora_config, run_dtype, device=device, vace_model=vace_model
            )

    @time_range
    def run(
        self,
        model_path: str | list[str],
        *,
        vace_model: list[str] | None = None,
        model_config: KsanaModelConfig = None,
        lora_config: None | list[list[KsanaLoraConfig]] = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
        comfy_bar_callback=None,
    ) -> list[KsanaModel]:
        log.info(f"{self.model_key} loading diffuion model from: {model_path}")
        load_model_path_or_files = self._valid_input_model_path(model_path)
        vace_model = self._valid_input_vace_model(vace_model, len(load_model_path_or_files))
        list_of_loras_list = self._valid_input_lora(lora_config, len(load_model_path_or_files))
        self.default_settings = load_default_settings(self.model_key, with_lora=list_of_loras_list is not None)
        device = device or torch.device("cuda")

        # TODO(rockcao): 检查pinned_memory在npu是否可用
        if KsanaDiffusionLoaderUnit._pinned_memory_manager is None and platform.is_gpu():
            KsanaDiffusionLoaderUnit._pinned_memory_manager = PinnedMemoryManager()
            log.info("Initialized shared PinnedMemoryManager for KsanaDiffusionLoaderUnit")

        res = []
        for i in range(len(load_model_path_or_files)):
            one_model_path = load_model_path_or_files[i]
            loras_list = list_of_loras_list[i] if list_of_loras_list is not None else None
            onve_vace_model = vace_model[i] if vace_model is not None else None
            model_state_dict = self._load_state_dict(
                one_model_path, model_config.run_dtype, device, loras_list, onve_vace_model
            )
            model_class = self._MAP_KEY_TO_MODEL_CLASS.get(self.model_key, None)
            if model_class is None:
                raise ValueError(f"model_key {self.model_key} not supported")
            model = model_class(
                self.model_key,
                model_config,
                dist_config,
                self.default_settings,
                pinned_memory_manager=KsanaDiffusionLoaderUnit._pinned_memory_manager,
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
            log.info(f"loading {self.model_key} to device:{device}, offload_device:{offload_device}")

            model.load(
                model_state_dict=model_state_dict,
                operations=operations,
                load_device=device,
                offload_device=offload_device,
            )
            log.debug(f"{self.model_key} model: {model.model}")
            model.load_state_dict(model_state_dict, strict=False)
            model.enable_only_infer()
            model.prepare_distributed_model(shard_fn)
            model.apply_dynamic_fp8_quant(
                linear_backend=model_config.linear_backend,
                load_device=device,
                model_state_dict=model_state_dict,
            )
            model.apply_torch_compile(model_config.torch_compile_config)
            # Note: apply_pinned_memory must be called after apply_torch_compile
            model.apply_pinned_memory(offload_device)

            if offload_device is not None:
                model = model.to(offload_device)
            res.append(model)
            if comfy_bar_callback is not None:
                comfy_bar_callback()
        return res[0] if len(res) == 1 else res
