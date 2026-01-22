import os

import torch

from ..config import KsanaLoraConfig, KsanaModelConfig
from ..models import KsanaModel, KsanaQwenImageModel, KsanaWanModel
from ..models.model_key import KsanaModelKey
from ..operations import build_ops
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import is_file_or_dir, log
from ..utils.lora import load_state_dict_and_merge_lora
from ..utils.profile import time_range


@KsanaUnitFactory.register(
    KsanaUnitType.LOADER, [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B, KsanaModelKey.QwenImage_T2I]
)
class KsanaDiffusionLoaderUnit(KsanaLoaderUnit):
    _MAP_KEY_TO_MODEL_CLASS = {
        KsanaModelKey.Wan2_2_I2V_14B: KsanaWanModel,
        KsanaModelKey.Wan2_2_T2V_14B: KsanaWanModel,
        KsanaModelKey.QwenImage_T2I: KsanaQwenImageModel,
    }

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

    def _valid_input_lora(self, lora_config: None | list[list[KsanaLoraConfig]]):
        if lora_config is None:
            return None
        if not isinstance(lora_config, (list, tuple)):
            raise ValueError(f"lora_config must be list of list of KsanaLoraConfig, but got {lora_config}")

        check_list = []
        if self.model_key in [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B]:
            if len(lora_config) != 2:
                raise ValueError(f"len of lora_config list must be 2 for {self.model_key}, but got {lora_config}")
            check_list = lora_config[0] + lora_config[1]
        else:
            if len(lora_config) != 1:
                raise ValueError(f"len of lora_config list must be 1 for {self.model_key}, but got {lora_config}")
            check_list = lora_config[0]
        if any(not isinstance(one, KsanaLoraConfig) for one in check_list):
            raise ValueError(f"lora_config must be list of KsanaLoraConfig, but got {lora_config}")
        return lora_config

    def _load_state_dict(
        self, model_path: str, run_dtype, device, lora_config: None | list[list[KsanaLoraConfig]] = None
    ):
        if self.model_key == KsanaModelKey.QwenImage_T2I and os.path.isdir(model_path):
            if getattr(self.default_settings.diffusion, "transformer_subdir", None) is None:
                raise ValueError(
                    f"transformer_subdir must be set in diffusion section of default_settings for"
                    f" {self.model_key}, but got {self.default_settings.diffusion}"
                )
            transformer_dir = os.path.join(model_path, self.default_settings.diffusion.transformer_subdir)
            return load_state_dict_and_merge_lora(transformer_dir, device=device)
        else:
            return load_state_dict_and_merge_lora(model_path, lora_config, run_dtype, device=device)

    @time_range
    def run(
        self,
        model_path: str | list[str],
        *,
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
        list_of_loras_list = self._valid_input_lora(lora_config)
        self.default_settings = load_default_settings(self.model_key, with_lora=list_of_loras_list is not None)
        device = device or torch.device("cuda")

        res = []
        for i in range(len(load_model_path_or_files)):
            one_model_path = load_model_path_or_files[i]
            loras_list = list_of_loras_list[i] if list_of_loras_list is not None else None
            model_state_dict = self._load_state_dict(one_model_path, model_config.run_dtype, device, loras_list)
            model_class = self._MAP_KEY_TO_MODEL_CLASS.get(self.model_key, None)
            if model_class is None:
                raise ValueError(f"model_key {self.model_key} not supported")
            model = model_class(self.model_key, model_config, dist_config, self.default_settings)
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
            model.do_load_state_dict(model_state_dict, strict=False)
            model.enable_only_infer()
            model.do_prepare_distributed_model(shard_fn)
            model.do_apply_dynamic_fp8_quant(linear_backend=model_config.linear_backend, load_device=device)
            model.do_apply_torch_compile(model_config.torch_compile_config)

            if offload_device is not None:
                model = model.to(offload_device)
            res.append(model)
            if comfy_bar_callback is not None:
                comfy_bar_callback()
        return res[0] if len(res) == 1 else res
