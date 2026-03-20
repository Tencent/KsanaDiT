# Copyright 2025 Tencent

"""Wan I2V GeneratorDef 注册。

Wan I2V 使用 WanDenoiseHandler（boundary 切换 + prepare_forward_kargs 含 y 参数）
和 WanLatentHandler（preprocess_base concat latent+mask + apply_aux_latent 噪声混合）。
"""

from kdit.generators.generator_def import GeneratorDef, register_generator_def
from kdit.generators.handlers.text_handler import TextHandler
from kdit.generators.handlers_impl.wan_denoise import WanDenoiseHandler
from kdit.generators.handlers_impl.wan_latent import WanLatentHandler
from kdit.models.model_key import ModelKey

register_generator_def(
    GeneratorDef(
        model_key=ModelKey.Wan2_2_I2V_14B,
        text_handler=TextHandler(),
        latent_handler=WanLatentHandler(model_key=ModelKey.Wan2_2_I2V_14B),
        denoise_handler=WanDenoiseHandler(),
    )
)
