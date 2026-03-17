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
from omegaconf import OmegaConf

from ..models.model_key import ModelKey

OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x))
_current_dir = os.path.dirname(os.path.abspath(__file__))


def _load_settings(config_path):
    conf = OmegaConf.load(config_path)
    if "_base_modules" in conf:
        for module_path in conf._base_modules:
            module_conf = _load_settings(os.path.join(_current_dir, module_path))
            conf = OmegaConf.merge(conf, module_conf)
    return conf


_MODEL_KEY_TO_CONF_PATH_MAP: dict = {
    ModelKey.Wan2_2_I2V_14B: "wan/i2v_14b.yaml",
    ModelKey.Wan2_2_T2V_14B: "wan/t2v_14b.yaml",
    ModelKey.Wan2_1_VACE_14B: "wan/vace_14b.yaml",
    ModelKey.VAE_WAN2_1: "wan/modules/vae_2_1.yaml",
    ModelKey.VAE_WAN2_2: "wan/modules/vae_2_2.yaml",
    ModelKey.T5TextEncoder: "wan/modules/t5_encoder.yaml",
    ModelKey.QwenImage_T2I: "qwen/t2i_20b.yaml",
    ModelKey.QwenImage_Edit: "qwen/edit_20b.yaml",
    ModelKey.QwenImageVAE: "qwen/modules/vae.yaml",
    ModelKey.Qwen2VLTextEncoder: "qwen/modules/text_encoder.yaml",
    ModelKey.Qwen2VLTextEncoderMultimodal: "qwen/modules/text_encoder_multimodal.yaml",
}

_MODEL_LORA_PATH_MAP = {
    ModelKey.Wan2_2_I2V_14B: "wan/modules/lora.yaml",
    ModelKey.Wan2_2_T2V_14B: "wan/modules/lora.yaml",
    ModelKey.Wan2_1_VACE_14B: "wan/modules/lora.yaml",
    ModelKey.QwenImage_T2I: "qwen/modules/lora.yaml",
    ModelKey.QwenImage_Edit: "qwen/modules/lora.yaml",
}

_PIPELINE_KEY_REGISTERED = False


def _ensure_pipeline_key_entries():
    """延迟注册 PipelineKey 条目，避免 settings ↔ pipelines 循环导入。"""
    global _PIPELINE_KEY_REGISTERED
    if _PIPELINE_KEY_REGISTERED:
        return
    _PIPELINE_KEY_REGISTERED = True

    from ..pipelines.pipeline_key import PipelineKey

    _MODEL_KEY_TO_CONF_PATH_MAP.update(
        {
            PipelineKey.Wan2_2_T2V_14B: "wan/t2v_14b.yaml",
            PipelineKey.Wan2_2_I2V_14B: "wan/i2v_14b.yaml",
            PipelineKey.Wan2_1_VACE_14B: "wan/vace_14b.yaml",
            PipelineKey.QwenImage_T2I: "qwen/t2i_20b.yaml",
            PipelineKey.QwenImage_Edit: "qwen/edit_20b.yaml",
        }
    )


def load_default_settings(model_key, with_lora: bool = False):
    """加载默认配置。model_key 可以是 ModelKey 或 PipelineKey。"""
    _ensure_pipeline_key_entries()
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
