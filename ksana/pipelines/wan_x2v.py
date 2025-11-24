from .x2v import KsanaX2VPipeline, KsanaDefaultArgs
import os
from ..models import KsanaDiffusionModel, KsanaT5Encoder, KsanaVAE
from dataclasses import dataclass, field
from ..cache import DCacheConfig


@dataclass(frozen=True)
class Wan2_2DefaultArgs(KsanaDefaultArgs):
    steps: int = field(default=50)
    cfg_scale: float = field(default=None)
    sample_shift: float = field(default=None)
    sample_solver: str = field(default="uni_pc")


@dataclass(frozen=True)
class WanLightLoraDefaultArgs(Wan2_2DefaultArgs):
    steps: int = field(default=4)
    cfg_scale: float | tuple[float, float] = field(default=(1.0, 1.0))
    sample_shift: float = field(default=5.0)
    sample_solver: str = field(default="euler")


class KsanaWanX2VPipeline(KsanaX2VPipeline):
    def __init__(self, model_version, task_type, default_model_config):
        """_summary_

        Args:
            model_version (_type_): "2.2" or "2.1"
            task_type (_type_): t2v, itv, v2v, etc.
        """
        super().__init__(task_type, default_model_config)
        self.model_version = model_version
        assert self.model_version in [
            "2.2",
            "2.1",
        ], f"model_version must be '2.2' or '2.1', but got {self.model_version}"
        assert default_model_config is not None, "default_model_config must be provided"
        assert (
            self.default_model_config.model_name == "wan2.2"
        ), f"model_name {self.default_model_config.model_name} is not supported"
        self.default_args = Wan2_2DefaultArgs()  # more could be 2.1 args when support more model versions

        assert self.default_model_config.task_type in [
            "t2v",
            "t2i",
            "v2v",
        ], f"task_type {self.default_model_config.task_type} is not supported"
        assert self.default_model_config.model_size in [
            "A14B",
            "5B",
        ], f"model_size {self.default_model_config.model_size} is not supported"

    def load_text_encoder(self, checkpoint_dir, shard_fn):
        self.text_encoder = KsanaT5Encoder(self.default_model_config, checkpoint_dir=checkpoint_dir, shard_fn=shard_fn)
        return self.text_encoder

    def load_vae(self, checkpoint_dir, device):
        vae_type = "wan2_1" if self.task_type == "t2v" else "wan2_2"
        self.vae = KsanaVAE(
            vae_type=vae_type,
            model_config=self.default_model_config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )
        return self.vae

    def load_diffusion_model(
        self,
        checkpoint_dir,
        lora_dir=None,
        torch_compile_config=None,
        dist_config=None,
        shard_fn=None,
        device=None,
        offload_device=None,
    ):
        if lora_dir is not None:
            self.default_args = WanLightLoraDefaultArgs()
        # import ipdb; ipdb.set_trace()
        if self.model_version == "2.2":
            high_noise_model = KsanaDiffusionModel(self.default_model_config, dist_config)
            high_lora_dir = (
                os.path.join(lora_dir, self.default_model_config.high_noise_lora_checkpoint)
                if lora_dir is not None
                else None
            )
            high_noise_model.load(
                checkpoint_dir=checkpoint_dir,
                subfolder=self.default_model_config.high_noise_checkpoint,
                lora_dir=high_lora_dir,
                torch_compile_config=torch_compile_config,
                shard_fn=shard_fn,
            )
            if offload_device is not None:
                high_noise_model.to(offload_device)
            low_noise_model = KsanaDiffusionModel(self.default_model_config, dist_config)
            low_lora_dir = (
                os.path.join(lora_dir, self.default_model_config.low_noise_lora_checkpoint)
                if lora_dir is not None
                else None
            )
            low_noise_model.load(
                checkpoint_dir=checkpoint_dir,
                subfolder=self.default_model_config.low_noise_checkpoint,
                lora_dir=low_lora_dir,
                torch_compile_config=torch_compile_config,
                shard_fn=shard_fn,
            )
            if offload_device is not None:
                low_noise_model.to(offload_device)
            self.model = (high_noise_model, low_noise_model)
        else:
            self.model = KsanaDiffusionModel(self.default_model_config, dist_config)
            self.model.load(
                checkpoint_dir=checkpoint_dir,
                # subfolder=self.default_model_config.high_noise_checkpoint,
                lora_dir=lora_dir,
                torch_compile_config=torch_compile_config,
                shard_fn=shard_fn,
            )
        return self.model

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

    def forward_diffusion_model(self, positive, negative, device=None, offload_device=None, **kwargs):
        sample_config = self.process_input_sample_config(**kwargs)
        runtime_config = self.process_input_runtime_config(**kwargs)
        high_cache_config, low_cache_config = self.process_input_cache(kwargs.get("cache_method", None))

        latents = self.generate_video_with_tensors(
            model=self.model,
            positive=positive,
            negative=negative,
            latents=None,
            sample_config=sample_config,
            runtime_config=runtime_config,
            high_cache_config=high_cache_config,
            low_cache_config=low_cache_config,
            device=device,
            offload_device=offload_device,
        )
        del positive, negative
        return latents
