# Copyright 2025 Tencent

"""Wan VACE GeneratorDef 注册。

VACE 使用 VaceDenoiseHandler（继承 Wan boundary + bidirectional/video_control/post_run）
和 WanLatentHandler（apply_aux_latent 直通 + validate_noise_shape）。
"""

from kdit.generators.generator_def import GeneratorDef, register_generator_def
from kdit.generators.handlers.text_handler import TextHandler
from kdit.generators.handlers_impl.vace_denoise import VaceDenoiseHandler
from kdit.generators.handlers_impl.wan_latent import WanLatentHandler
from kdit.models.model_key import ModelKey

register_generator_def(
    GeneratorDef(
        model_key=ModelKey.Wan2_1_VACE_14B,
        text_handler=TextHandler(),
        latent_handler=WanLatentHandler(model_key=ModelKey.Wan2_1_VACE_14B),
        denoise_handler=VaceDenoiseHandler(),
    )
)
