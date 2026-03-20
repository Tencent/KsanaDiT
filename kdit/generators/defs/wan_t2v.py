# Copyright 2025 Tencent

"""Wan T2V GeneratorDef 注册。

Wan T2V 使用 WanDenoiseHandler（boundary 切换 + prepare_forward_kargs）
和 WanLatentHandler（apply_aux_latent 直通 + validate_noise_shape）。
"""

from kdit.generators.generator_def import GeneratorDef, register_generator_def
from kdit.generators.handlers.text_handler import TextHandler
from kdit.generators.handlers_impl.wan_denoise import WanDenoiseHandler
from kdit.generators.handlers_impl.wan_latent import WanLatentHandler
from kdit.models.model_key import ModelKey

register_generator_def(
    GeneratorDef(
        model_key=ModelKey.Wan2_2_T2V_14B,
        text_handler=TextHandler(),
        latent_handler=WanLatentHandler(model_key=ModelKey.Wan2_2_T2V_14B),
        denoise_handler=WanDenoiseHandler(),
    )
)
