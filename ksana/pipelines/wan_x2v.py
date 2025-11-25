from .x2v import KsanaX2VPipeline, KsanaDefaultArgs
import os
from ..models import KsanaDiffusionModel, KsanaT5Encoder, KsanaVAE
from dataclasses import dataclass, field
from ..cache import DCacheConfig
from ..config import KsanaSampleConfig, KsanaRuntimeConfig, KsanaPipelineConfig, KsanaModelConfig
from ..utils.profile import time_range


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
    def load_diffusion_model_from_pretrained(
        self,
        checkpoint_dir,
        lora_dir=None,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        shard_fn=None,
        device=None,
        offload_device=None,
    ):
        if lora_dir is not None:
            self.default_args = WanLightLoraDefaultArgs()
        if self.pipeline_config.model_name == "wan2.2":
            high_noise_model = KsanaDiffusionModel(model_config, self.pipeline_config, dist_config)
            high_lora_dir = (
                os.path.join(lora_dir, self.pipeline_config.default_config.high_noise_lora_checkpoint)
                if lora_dir is not None
                else None
            )
            high_noise_model.load(
                checkpoint_dir=checkpoint_dir,
                subfolder=self.pipeline_config.default_config.high_noise_checkpoint,
                lora_dir=high_lora_dir,
                shard_fn=shard_fn,
            )
            if offload_device is not None:
                high_noise_model.to(offload_device)
            low_noise_model = KsanaDiffusionModel(model_config, self.pipeline_config, dist_config)
            low_lora_dir = (
                os.path.join(lora_dir, self.pipeline_config.default_config.low_noise_lora_checkpoint)
                if lora_dir is not None
                else None
            )
            low_noise_model.load(
                checkpoint_dir=checkpoint_dir,
                subfolder=self.pipeline_config.default_config.low_noise_checkpoint,
                lora_dir=low_lora_dir,
                shard_fn=shard_fn,
            )
            if offload_device is not None:
                low_noise_model.to(offload_device)
            self.model = (high_noise_model, low_noise_model)
        else:
            self.model = KsanaDiffusionModel(model_config, self.pipeline_config, dist_config)
            self.model.load(
                checkpoint_dir=checkpoint_dir,
                lora_dir=lora_dir,
                shard_fn=shard_fn,
            )
        return self.model

    @time_range
    def load_diffusion_model_from_comfy(
        self,
        model_config: KsanaModelConfig = None,
        dist_config=None,
        comfy_model_path: str = None,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_operations=None,
        device=None,
        offload_device=None,
        shard_fn=None,
    ):
        model = KsanaDiffusionModel(model_config, self.pipeline_config, dist_config)
        model.load(
            comfy_model_path=comfy_model_path,
            comfy_model_config=comfy_model_config,
            comfy_model_state_dict=comfy_model_state_dict,
            comfy_operations=comfy_operations,
            load_device=device,
            offload_device=offload_device,
            shard_fn=shard_fn,
        )
        return model

    def process_input_cache(self, cache_method):
        high_cache_config = None
        low_cache_config = None
        if cache_method == "DCache":
            high_cache_config = DCacheConfig(
                fast_degree=70,
                slow_degree=35,
                fast_force_calc_every_n_steps=1,
                slow_force_calc_every_n_steps=5,
                name="high_dcache",
            )
            low_cache_config = DCacheConfig(
                fast_degree=65,
                slow_degree=25,
                fast_force_calc_every_n_steps=2,
                slow_force_calc_every_n_steps=4,
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
        high_cache_config, low_cache_config = self.process_input_cache(runtime_config.cache_method)

        latents = self.generate_video_with_tensors(
            model=self.model,
            positive=positive,
            negative=negative,
            latents=None,
            sample_config=self.prepare_sample_default_args(sample_config),
            runtime_config=KsanaRuntimeConfig.copy_with_default(runtime_config, self.pipeline_config.default_config),
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
            device=device,
            offload_device=offload_device,
        )
        del positive, negative
        return latents
