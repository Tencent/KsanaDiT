from easydict import EasyDict

from ksana.config.sample_config import KsanaSolverBackend

qwen_image_t2i = EasyDict(__name__="Config: Qwen-Image T2I")

qwen_image_t2i.text_encoder_subdir = "text_encoder"
qwen_image_t2i.tokenizer_subdir = "tokenizer"
qwen_image_t2i.transformer_subdir = "transformer"
qwen_image_t2i.vae_subdir = "vae"
qwen_image_t2i.scheduler_subdir = "scheduler"

qwen_image_t2i.size = (1024, 1024)
qwen_image_t2i.aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

qwen_image_t2i.text_len = 1024
qwen_image_t2i.prompt_template = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
qwen_image_t2i.prompt_template_drop_idx = 34
qwen_image_t2i.text_dtype = "bfloat16"

qwen_image_t2i.vae_stride = (1, 8, 8)
qwen_image_t2i.z_dim = 16
qwen_image_t2i.vae_scale_factor = 8
qwen_image_t2i.num_heads = 24

qwen_image_t2i.patch_size = 2
qwen_image_t2i.in_channels = 64
qwen_image_t2i.out_channels = 16
qwen_image_t2i.num_layers = 60
qwen_image_t2i.num_attention_heads = 24
qwen_image_t2i.attention_head_dim = 128
qwen_image_t2i.joint_attention_dim = 3584
qwen_image_t2i.axes_dims_rope = (16, 56, 56)
qwen_image_t2i.guidance_embeds = False

qwen_image_t2i.num_train_timesteps = 1000
qwen_image_t2i.sample_steps = 50
qwen_image_t2i.sample_guide_scale = 4.0
qwen_image_t2i.sample_neg_prompt = " "
qwen_image_t2i.sample_solver = KsanaSolverBackend.FLOWMATCH_EULER

qwen_image_t2i.base_seq_len = 256
qwen_image_t2i.max_seq_len = 4096
qwen_image_t2i.base_shift = 0.5
qwen_image_t2i.max_shift = 1.15

__all__ = ["qwen_image_t2i"]
