
from .simply_test import DITConfig

NODE_CLASS_MAPPINGS = {
    "VideoGenerator": VideoGenerator,
    "InferenceArgs": InferenceArgs,
    "VAEConfig": VAEConfig,
    "TextEncoderConfig": TextEncoderConfig,
    "DITConfig": DITConfig,
    "LoadImagePath": LoadImagePath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGenerator": "Video Generator",
    "InferenceArgs": "Inference Args",
    "VAEConfig": "VAE Config",
    "TextEncoderConfig": "Text Encoder Config",
    "DITConfig": "DIT Config",
    "LoadImagePath": "Load Image Path"
}