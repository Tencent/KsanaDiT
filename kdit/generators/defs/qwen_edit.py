# Copyright 2025 Tencent

"""Qwen Edit GeneratorDef 注册。

Qwen Edit 使用 QwenTextHandler（tuple conditioning）、
QwenLatentHandler（patchify/unpatchify）和
QwenDenoiseHandler（prepare_forward + norm rescale CFG）。
"""

from kdit.generators.generator_def import GeneratorDef, register_generator_def
from kdit.generators.handlers_impl.qwen_denoise import QwenDenoiseHandler
from kdit.generators.handlers_impl.qwen_latent import QwenLatentHandler
from kdit.generators.handlers_impl.qwen_text import QwenTextHandler
from kdit.models.model_key import ModelKey

_qwen_latent = QwenLatentHandler()

register_generator_def(
    GeneratorDef(
        model_key=ModelKey.QwenImage_Edit,
        text_handler=QwenTextHandler(),
        latent_handler=_qwen_latent,
        denoise_handler=QwenDenoiseHandler(
            model_key=ModelKey.QwenImage_Edit,
            latent_handler=_qwen_latent,
        ),
    )
)
