import os
from abc import ABC
from ..utils import log, time_range  # , vProfiler

from .wan.model import WanModel
from .wan.configs import WAN2_2_CONFIGS
import time


def create_ksana_model(model_path, comfy_model_config):
    model_name = os.path.basename(model_path)
    model_type = KsanaModel.get_model_type(comfy_model_config, model_name)  # t2v
    model_kind = KsanaModel.get_model_kind(model_name)  # wan2.2
    model_size = KsanaModel.get_model_size(model_kind, model_name)  # 14b
    assert model_kind in ["wan2.2"], "only support wan2.2 yet"

    return KsanaModel(model_kind, model_type, model_size)


class KsanaModel(ABC):
    def __init__(self, model_kind, model_type, model_size):
        self.model_kind = model_kind  # wan2.2
        self.task_type = model_type  # t2v
        self.model_size = model_size  # 14b

    @time_range
    def load(
        self,
        model_path,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_model_options: dict = None,
        disable_weight_init_operations=None,
        dtype=None,
        load_device=None,
        offload_device=None,
    ):
        assert self.model_kind in ["wan2.2"], "only support wan2.2 yet"
        if "wan" in self.model_kind:
            self.load_wan_model(
                model_path,
                comfy_model_config,
                comfy_model_state_dict,
                comfy_model_options,
                disable_weight_init_operations,
                dtype,
                load_device,
                offload_device,
            )
        else:
            raise ValueError(f"model_kind {self.model_kind} not supported")

    def load_model_weights(self, model, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix) :]] = sd.pop(k)

        to_load = model.model_config.process_unet_state_dict(to_load)
        m, u = model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            log.warning("unet missing: {}".format(m))

        if len(u) > 0:
            log.warning("unet unexpected: {}".format(u))
        del to_load
        return model

    def load_wan_model(
        self,
        model_path,
        comfy_model_config: dict = None,
        comfy_model_state_dict=None,
        comfy_model_options: dict = None,
        disable_weight_init_operations=None,
        dtype=None,
        load_device=None,
        offload_device=None,
    ):
        log.info(
            f"model_path:{model_path}, comfy_model_config:{comfy_model_config}, "
            f"load_device:{load_device}, offload_device:{offload_device}, comfy_model_options:{comfy_model_options}"
        )
        config_type = f"{self.task_type}-{self.model_size}"
        log.info(f"config_type: {config_type}")
        self._default_config = WAN2_2_CONFIGS[config_type]
        in_dim = comfy_model_config.get("in_dim", self.default_config.get("in_dim", 16))
        out_dim = comfy_model_config.get("out_dim", self.default_config.get("out_dim", 16))
        self.run_dtype = dtype

        # with vProfiler("KsanaModel.load_wan_model"):
        if comfy_model_state_dict is not None:
            start = time.time()
            self.model = WanModel(
                model_type=self.task_type,
                patch_size=self.default_config.patch_size,
                text_len=self.default_config.text_len,
                in_dim=in_dim,
                dim=self.default_config.dim,
                ffn_dim=self.default_config.ffn_dim,
                freq_dim=self.default_config.freq_dim,
                out_dim=out_dim,
                num_heads=self.default_config.num_heads,
                num_layers=self.default_config.num_layers,
                window_size=self.default_config.window_size,
                qk_norm=self.default_config.qk_norm,
                cross_attn_norm=self.default_config.cross_attn_norm,
                eps=self.default_config.eps,
                disable_weight_init_operations=disable_weight_init_operations,
                device=offload_device,
                dtype=dtype,
            )
            stop1 = time.time()
            self.model.load_state_dict(comfy_model_state_dict, strict=False)
            stop2 = time.time()
            log.info(f"create model takes: {(stop1 - start):.2f}, load states takes {(stop2 - stop1):.2f} seconds")
            log.info(f"model: {self.model}")
        else:
            # TODO: add ut to test this branch
            self.model = WanModel.from_pretrained(model_path)

        # self.high_noise_model = self._configure_model(
        #     model=self.high_noise_model,
        #     use_sp=use_sp,
        #     dit_fsdp=dit_fsdp,
        #     shard_fn=shard_fn,
        #     convert_model_dtype=convert_model_dtype)

        # def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
        #                      convert_model_dtype):
        #     """
        #     Configures a model object. This includes setting evaluation modes,
        #     applying distributed parallel strategy, and handling device placement.

        #     Args:
        #         model (torch.nn.Module):
        #             The model instance to configure.
        #         use_sp (`bool`):
        #             Enable distribution strategy of sequence parallel.
        #         dit_fsdp (`bool`):
        #             Enable FSDP sharding for DiT model.
        #         shard_fn (callable):
        #             The function to apply FSDP sharding.
        #         convert_model_dtype (`bool`):
        #             Convert DiT model parameters dtype to 'config.param_dtype'.
        #             Only works without FSDP.

        #     Returns:
        #         torch.nn.Module:
        #             The configured model.
        #     """
        #     model.eval().requires_grad_(False)

        #     if use_sp:
        #         for block in model.blocks:
        #             block.self_attn.forward = types.MethodType(
        #                 sp_attn_forward, block.self_attn)
        #         model.forward = types.MethodType(sp_dit_forward, model)

        #     if dist.is_initialized():
        #         dist.barrier()

        #     if dit_fsdp:
        #         model = shard_fn(model)
        #     else:
        #         if convert_model_dtype:
        #             model.to(self.param_dtype)
        #         if not self.init_on_cpu:
        #             model.to(self.device)

        #     return model

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    @property
    def default_config(self):  #: change to default_model_config
        return self._default_config

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_model_type(input_model_config: dict, model_name: str):
        """
        Get the model type from the state dict.
        return t2v or s2v or i2v or ti2v
        """
        choise = ["t2v", "s2v", "i2v", "ti2v"]
        model_type = input_model_config.get("model_type", None)
        lower_model_name = model_name.lower()
        if model_type is not None:
            if model_type in choise:
                return model_type
            else:
                raise ValueError(f"model_type {model_type} not in {choise}")
        else:
            for t in choise:
                if t in lower_model_name:
                    return t
        log.info(f"model_type: {model_type}")

        raise ValueError(f"Can not detect model_type from model_config:{input_model_config}, model_name:{model_name}")
        # f"{wan, flux, xxx}-{version:2.2}-{task_type:t2v}-{params:14B}"

    @staticmethod
    def get_model_kind(model_name: str):
        """
        Get the model kind from the model name.
        return wan or flux
        """
        wan22 = ["wan2.2", "wan22", "wan2_2"]
        lower_model_name = model_name.lower()
        for k in wan22:
            if k in lower_model_name:
                return "wan2.2"
        # wan21 = ["wan2.1", "wan21", "wan2_1"]
        # for k in wan21:
        #     if k in model_name.lower():
        #         return "wan2.1"
        raise ValueError(f"model_name {model_name} not supported")

    @staticmethod
    def get_model_size(model_kind: str, model_name: str):
        """
        Get the model size from the model_kind and model name.
        return 14b, 1.3b, or others
        """
        lower_model_name = model_name.lower()
        if model_kind == "wan2.2":
            if "14b" in lower_model_name:
                return "A14B"
            elif "5b" in lower_model_name:
                return "5B"
        elif model_kind == "wan2.1":
            if "14b" in lower_model_name:
                return "14B"
            elif "1.3b" in lower_model_name or "1_3b" in lower_model_name:
                return "1.3B"
        else:
            raise ValueError(f"can not detect model_size from model_kind:{model_kind}, model_name:{model_name}")
