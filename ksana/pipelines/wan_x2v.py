from .x2v import KsanaX2VPipeline, KsanaDefaultArgs
import os
from ..models import KsanaWanModel, KsanaT5Encoder, KsanaVAE
from dataclasses import dataclass, field
from ..cache import DCacheConfig
from ..config import KsanaSampleConfig, KsanaRuntimeConfig, KsanaPipelineConfig, KsanaModelConfig
from ..utils.profile import time_range
from ..utils.logger import log
from ..utils import is_dir


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
        assert pipeline_config.model_name in [
            "wan2.2",
            "wan2.1",
        ], f"model_name must be 'wan2.2' or 'wan2.1', but got {pipeline_config.model_name}"
        assert pipeline_config.model_config is not None, "model_config must be provided"
        assert pipeline_config.default_config is not None, "default_config must be provided"
        assert pipeline_config.model_name == "wan2.2", f"model_name {pipeline_config.model_name} is not supported"
        self.default_args = Wan2_2DefaultArgs()  # more could be 2.1 args when support more model versions

        assert self.pipeline_config.task_type in [
            "t2v",
            "t2i",
            "v2v",
        ], f"task_type {self.pipeline_config.task_type} is not supported"
        assert self.pipeline_config.model_size in [
            "A14B",
            "5B",
        ], f"model_size {self.pipeline_config.model_size} is not supported"

    def load_text_encoder(self, checkpoint_dir, shard_fn):
        self.text_encoder = KsanaT5Encoder(
            self.pipeline_config.default_config, checkpoint_dir=checkpoint_dir, shard_fn=shard_fn
        )
        return self.text_encoder

    def load_vae(self, checkpoint_dir, device):
        vae_type = "wan2_1" if self.pipeline_config.task_type == "t2v" else "wan2_2"
        self.vae = KsanaVAE(
            vae_type=vae_type,
            default_pipeline_config=self.pipeline_config.default_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        return self.vae

    @time_range
    def load_one_diffusion_model(
        self,
        model_path,
        *,
        lora_file=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        comfy_model_config=None,
        comfy_model_state_dict=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        if lora_file is not None:
            self.default_args = WanLightLoraDefaultArgs()
        model = KsanaWanModel(model_config, self.pipeline_config, dist_config)
        model.load(
            model_path=model_path,
            lora_file=lora_file,
            comfy_model_config=comfy_model_config,
            comfy_model_state_dict=comfy_model_state_dict,
            load_device=device,
            offload_device=offload_device,
            shard_fn=shard_fn,
        )
        return model

    @time_range
    def load_diffusion_model(
        self,
        model_path,
        *,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        comfy_model_config=None,
        comfy_model_state_dict=None,
        dist_config=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        log.info(f"load_model_path_or_files: {model_path}")
        load_model_path_or_files = model_path
        load_lora_files = lora_dir

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
            if lora_dir is not None:
                load_lora_files = [
                    os.path.join(lora_dir, self.pipeline_config.default_config.high_noise_lora_checkpoint),
                    os.path.join(lora_dir, self.pipeline_config.default_config.low_noise_lora_checkpoint),
                ]
        else:
            if not os.path.isfile(model_path):
                raise ValueError(f"model_path {model_path} not exist, or file must be a safetensors file")

        if isinstance(load_model_path_or_files, (list, tuple)):
            res = []
            for i in range(len(load_model_path_or_files)):
                one_model_path = load_model_path_or_files[i]
                one_lora_file = load_lora_files[i] if load_lora_files is not None else None
                one_model = self.load_one_diffusion_model(
                    model_path=one_model_path,
                    lora_file=one_lora_file,
                    model_config=model_config,
                    comfy_model_config=comfy_model_config,
                    comfy_model_state_dict=comfy_model_state_dict,
                    dist_config=dist_config,
                    shard_fn=shard_fn,
                    device=device,
                    offload_device=offload_device,
                )
                if offload_device is not None:
                    one_model = one_model.to(offload_device)
                res.append(one_model)
            return res
        else:
            return self.load_one_diffusion_model(
                model_path=load_model_path_or_files,
                lora_file=load_lora_files,
                model_config=model_config,
                comfy_model_config=comfy_model_config,
                comfy_model_state_dict=comfy_model_state_dict,
                dist_config=dist_config,
                shard_fn=shard_fn,
                device=device,
                offload_device=offload_device,
            )

    def process_input_cache(self, cache_method):
        high_cache_config = None
        low_cache_config = None
        if cache_method == "DCache":
            high_cache_config = DCacheConfig(
                fast_degree=70,
                slow_degree=35,
                fast_force_calc_every_n_step=1,
                slow_force_calc_every_n_step=5,
                name="high_dcache",
            )
            low_cache_config = DCacheConfig(
                fast_degree=65,
                slow_degree=25,
                fast_force_calc_every_n_step=2,
                slow_force_calc_every_n_step=4,
                name="low_dcache",
            )
        return high_cache_config, low_cache_config

    def forward_diffusion_model(
        self,
        positive,
        negative,
        sample_config: KsanaSampleConfig,
        runtime_config: KsanaRuntimeConfig,
        device=None,
        offload_device=None,
    ):
        runtime_config = KsanaRuntimeConfig.copy_with_default(runtime_config, self.pipeline_config.default_config)
        high_cache_config, low_cache_config = self.process_input_cache(runtime_config.cache_method)

        latents = self.generate_video_with_tensors(
            model=self.diffusion_model,
            positive=positive,
            negative=negative,
            latents=None,
            sample_config=self.prepare_sample_default_args(sample_config),
            runtime_config=runtime_config,
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
            device=device,
            offload_device=offload_device,
        )
        del positive, negative

        self.offload_diffusion_model_to(runtime_config.offload_model, offload_device)
        return latents
