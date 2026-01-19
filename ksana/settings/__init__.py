import os

import torch
from omegaconf import OmegaConf

from ..models.model_key import KsanaModelKey

OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x))
_current_dir = os.path.dirname(os.path.abspath(__file__))


def _load_settings(config_path):
    conf = OmegaConf.load(config_path)
    if "_base_modules" in conf:
        for module_path in conf._base_modules:
            module_conf = _load_settings(os.path.join(_current_dir, module_path))
            conf = OmegaConf.merge(conf, module_conf)
    return conf


_MODEL_KEY_TO_CONF_PATH_MAP = {
    KsanaModelKey.Wan2_2_I2V_14B: "wan/i2v_14b.yaml",
    KsanaModelKey.Wan2_2_T2V_14B: "wan/t2v_14b.yaml",
    KsanaModelKey.VAE_WAN2_1: "wan/modules/vae_2_1.yaml",
    KsanaModelKey.VAE_WAN2_2: "wan/modules/vae_2_2.yaml",
    KsanaModelKey.T5TextEncoder: "wan/modules/t5_encoder.yaml",
    KsanaModelKey.QwenImage_T2I: "qwen/t2i_20b.yaml",
    KsanaModelKey.QwenImageVAE: "qwen/modules/vae.yaml",
    KsanaModelKey.Qwen2VLTextEncoder: "qwen/modules/text_encoder.yaml",
}

_MODEL_LORA_PATH_MAP = {
    KsanaModelKey.Wan2_2_I2V_14B: "wan/modules/lora.yaml",
    KsanaModelKey.Wan2_2_T2V_14B: "wan/modules/lora.yaml",
    KsanaModelKey.QwenImage_T2I: "qwen/modules/lora.yaml",
}


def load_default_settings(model_key: KsanaModelKey, with_lora: bool = False):
    if model_key not in _MODEL_KEY_TO_CONF_PATH_MAP:
        raise ValueError(f"model_key {model_key} does not have default settings yet!")
    conf = _load_settings(os.path.join(_current_dir, _MODEL_KEY_TO_CONF_PATH_MAP[model_key]))

    if with_lora:
        if model_key in _MODEL_LORA_PATH_MAP:
            conf = OmegaConf.merge(conf, _load_settings(os.path.join(_current_dir, _MODEL_LORA_PATH_MAP[model_key])))
        else:
            raise ValueError(f"model_key {model_key} does not have lora settings yet!")
    return conf


__all__ = ["load_default_settings"]
