import os
from pathlib import Path

from ..config import KsanaModelConfig
from ..models import KsanaModel, KsanaQwenImageModel, KsanaWanModel
from ..models.model_key import KsanaModelKey
from ..settings import load_default_settings
from ..units import KsanaLoaderUnit, KsanaUnitFactory, KsanaUnitType
from ..utils import is_file_or_dir, log
from ..utils.lora import load_state_dict_and_merge_lora
from ..utils.profile import time_range


class KsanaDiffusionLoader(KsanaLoaderUnit):

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

    def _valid_input_lora(self, lora: None | list[list[dict]]):
        if lora is None:
            return None
        if isinstance(lora, str) and not Path(lora).is_file():
            raise ValueError(f"lora {lora} must be a file")

        if not isinstance(lora, (list, tuple)):
            raise ValueError(f"lora must be list of dict or files, but got {lora}")
        return lora


@KsanaUnitFactory.register(KsanaUnitType.LOADER, [KsanaModelKey.Wan2_2_I2V_14B, KsanaModelKey.Wan2_2_T2V_14B])
class KsanaWanVideoLoader(KsanaDiffusionLoader):

    @time_range
    def run(
        self,
        model_path: str | list[str],
        *,
        lora: None | list[list[dict]] = None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
        comfy_bar_callback=None,
    ) -> list[KsanaModel]:
        log.info(f"load_model_from: {model_path}")
        load_model_path_or_files = self._valid_input_model_path(model_path)
        list_of_loras_list = self._valid_input_lora(lora)
        self.default_settings = load_default_settings(self.model_key, with_lora=list_of_loras_list is not None)

        res = []
        for i in range(len(load_model_path_or_files)):
            one_model_path = load_model_path_or_files[i]
            loras_list = list_of_loras_list[i] if list_of_loras_list is not None else None
            model_state_dict = load_state_dict_and_merge_lora(
                one_model_path, loras_list, model_config.run_dtype, device=device
            )
            model = KsanaWanModel(self.model_key, model_config, dist_config, self.default_settings)
            model.load(
                model_state_dict=model_state_dict,
                load_device=device,
                offload_device=offload_device,
                shard_fn=shard_fn,
            )
            if offload_device is not None:
                model = model.to(offload_device)
            res.append(model)
            if comfy_bar_callback is not None:
                comfy_bar_callback()
        return res[0] if len(res) == 1 else res


@KsanaUnitFactory.register(KsanaUnitType.LOADER, KsanaModelKey.QwenImage_T2I)
class KsanaQwenImageLoader(KsanaDiffusionLoader):
    @time_range
    def run(
        self,
        model_path,
        *,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        super().run()
        model = KsanaQwenImageModel(model_config, self.default_settings, dist_config)
        load_device = str(device) if device is not None else "cuda"
        default_cfg = self.pipeline_config.default_config  # TODO(TJ): remove
        if os.path.isfile(model_path):
            state_dict = load_state_dict_and_merge_lora(model_path, device=load_device)
        else:
            transformer_dir = os.path.join(model_path, default_cfg.transformer_subdir)
            state_dict = load_state_dict_and_merge_lora(transformer_dir, device=load_device)

        model.load(
            model_state_dict=state_dict,
            input_model_config=default_cfg,
            load_device=device,
            offload_device=offload_device,
        )
        return model
