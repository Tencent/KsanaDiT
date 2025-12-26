from .x2v import KsanaX2VPipeline, KsanaDefaultArgs
import os
from ..models import KsanaWanModel, KsanaT5Encoder, KsanaVAE
from dataclasses import dataclass, field
from ..config import KsanaPipelineConfig, KsanaModelConfig
from ..utils.profile import time_range
from ..utils.logger import log
from ..utils import is_dir
from ..utils.lora import load_state_dict_and_merge_lora, build_loras_list

from ..models.model_key import WAN2_2, WAN2_1, X2V_TYPES
from ..models.base_model import KsanaModel


@dataclass(frozen=True)
class Wan2_2DefaultArgs(KsanaDefaultArgs):
    steps: int = field(default=50)
    sample_solver: str = field(default="uni_pc")


@dataclass(frozen=True)
class WanLightLoraDefaultArgs(Wan2_2DefaultArgs):
    steps: int = field(default=4)
    cfg_scale: float | tuple[float, float] = field(default=(1.0, 1.0))
    sample_shift: float = field(default=5.0)
    sample_solver: str = field(default="euler")


class KsanaWanX2VPipeline(KsanaX2VPipeline):
    def __init__(self, pipeline_config: KsanaPipelineConfig):
        """_summary_

        Args:
            pipeline_config (KsanaPipelineConfig): pipeline config
        """
        super().__init__(pipeline_config)
        assert (
            pipeline_config.model_name in WAN2_2 + WAN2_1
        ), f"model_name must be 'wan2.2' or 'wan2.1', but got {pipeline_config.model_name}"
        assert pipeline_config.model_config is not None, "model_config must be provided"
        assert pipeline_config.default_config is not None, "default_config must be provided"
        assert pipeline_config.model_name in WAN2_2, f"model_name {pipeline_config.model_name} is not supported"
        self.default_args = Wan2_2DefaultArgs()  # more could be 2.1 args when support more model versions

        assert (
            self.pipeline_config.task_type in X2V_TYPES
        ), f"task_type {self.pipeline_config.task_type} is not supported"
        assert self.pipeline_config.model_size in [
            "A14B",
            "5B",
        ], f"model_size {self.pipeline_config.model_size} is not supported"

    def load_text_encoder(self, checkpoint_dir, shard_fn):
        text_encoder = KsanaT5Encoder(
            self.pipeline_config.default_config, checkpoint_dir=checkpoint_dir, shard_fn=shard_fn
        )
        return text_encoder

    def load_vae(self, checkpoint_dir, device):
        model_path = os.path.join(checkpoint_dir, self.pipeline_config.default_config.vae_checkpoint)
        # wan2.2 vae use 2.1 vae at t2v and i2v
        vae_type = WAN2_1[0] if self.pipeline_config.task_type in ["t2v", "i2v"] else WAN2_2[0]
        return KsanaVAE(model_path=model_path, vae_type=vae_type, device=device)

    @time_range
    def load_one_diffusion_model(
        self,
        *,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        model_state_dict=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        model = KsanaWanModel(model_config, self.pipeline_config, dist_config)
        model.load(
            model_state_dict=model_state_dict,
            load_device=device,
            offload_device=offload_device,
            shard_fn=shard_fn,
        )
        if offload_device is not None:
            model = model.to(offload_device)
        return model

    @time_range
    def load_diffusion_model(
        self,
        model_path,
        *,
        lora: None | str | list[list[dict], list[dict]] = None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
        comfy_bar_callback=None,
    ) -> list[KsanaModel]:

        log.info(f"load_model_path_or_files: {model_path}")
        load_model_path_or_files = model_path

        list_of_loras_list = lora
        if isinstance(lora, str):
            lora_dir = lora
            list_of_loras_list = []
            list_of_loras_list.append(
                build_loras_list(os.path.join(lora_dir, self.pipeline_config.default_config.high_noise_lora_checkpoint))
            )
            list_of_loras_list.append(
                build_loras_list(os.path.join(lora_dir, self.pipeline_config.default_config.low_noise_lora_checkpoint))
            )
        if list_of_loras_list is not None:
            self.default_args = WanLightLoraDefaultArgs()

        # three cases:
        # 1. [local load] model_path is a list of safetensors files used in load fp8 safetensors
        # 2. [local load] model_path is a dir, used to compatibale with from_pretrain checkpoint
        # 3. [comfy load] model_path is a safetensors file
        if isinstance(model_path, (list, tuple)):
            for file in model_path:
                assert os.path.isfile(
                    file
                ), f"model_path must be list of files or {file} not exist, model_path:{model_path}"
        elif is_dir(model_path):
            if self.pipeline_config.model_name == "wan2.2":
                load_model_path_or_files = [
                    os.path.join(model_path, self.pipeline_config.default_config.high_noise_checkpoint),
                    os.path.join(model_path, self.pipeline_config.default_config.low_noise_checkpoint),
                ]
        else:
            if not os.path.isfile(model_path):
                raise ValueError(f"model_path {model_path} not exist, or file must be a safetensors file")

        if isinstance(load_model_path_or_files, (list, tuple)):
            is_high = [True, False]
            res = []
            for i in range(len(load_model_path_or_files)):
                one_model_path = load_model_path_or_files[i]
                loras_list = list_of_loras_list[i] if list_of_loras_list is not None else None
                model_state_dict = load_state_dict_and_merge_lora(one_model_path, loras_list, device=device)
                one_model = self.load_one_diffusion_model(
                    model_config=model_config,
                    model_state_dict=model_state_dict,
                    dist_config=dist_config,
                    shard_fn=shard_fn,
                    device=device,
                    offload_device=offload_device,
                )
                one_model.set_as_high_or_low(is_high[i])
                res.append(one_model)
                if comfy_bar_callback is not None:
                    comfy_bar_callback()
            return res
        else:
            loras_list = list_of_loras_list[0] if list_of_loras_list is not None else None
            model_state_dict = load_state_dict_and_merge_lora(load_model_path_or_files, loras_list, device=device)
            one_model = self.load_one_diffusion_model(
                model_config=model_config,
                model_state_dict=model_state_dict,
                dist_config=dist_config,
                shard_fn=shard_fn,
                device=device,
                offload_device=offload_device,
            )
            if comfy_bar_callback is not None:
                comfy_bar_callback()
            return [one_model]
