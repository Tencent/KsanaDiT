import os

import torch

import ksana.nodes as nodes
from ksana.config import KsanaAttentionBackend, KsanaAttentionConfig, KsanaLinearBackend, KsanaRadialSageAttentionConfig
from ksana.models.model_key import KsanaModelKey

IMG_SHAPE_T2V = [1, 16, 16, 32, 32]
IMG_SHAPE_I2V = [1, 20, 16, 32, 32]
IMG_SHAPE_T2I = [1, 16, 1, 32, 32]

TEST_STEPS = 2

WAN_TEXT_SHAPE = [1, 512, 4096]
QWEN_TEXT_SHAPE = [1, 1024, 3584]

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models"
COMFY_MODEL_DIFFUSION_ROOT = os.path.join(COMFY_MODEL_ROOT, "diffusion_models")

SEED = 321
RUN_DTYPE = torch.float16

TEST_MODELS = [
    ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", IMG_SHAPE_T2V, WAN_TEXT_SHAPE, KsanaModelKey.Wan2_2_T2V_14B),
    ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", IMG_SHAPE_I2V, WAN_TEXT_SHAPE, KsanaModelKey.Wan2_2_I2V_14B),
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", IMG_SHAPE_T2V, WAN_TEXT_SHAPE, KsanaModelKey.Wan2_2_T2V_14B),
    ("wan2.2_i2v_high_noise_14B_fp16.safetensors", IMG_SHAPE_I2V, WAN_TEXT_SHAPE, KsanaModelKey.Wan2_2_I2V_14B),
    ("qwen_image_2512_fp8_e4m3fn.safetensors", IMG_SHAPE_T2I, QWEN_TEXT_SHAPE, KsanaModelKey.QwenImage_T2I),
]

TEST_ONE_GPU_EPS_PLACE = 3
TEST_GPUS_EPS_PLACE = 3


def run_load_and_generate(model_path, image_latent_shape, text_shape, steps, **kwargs):
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(SEED)
    positive_text_embeddings = torch.randn(
        *text_shape,
        dtype=RUN_DTYPE,
        device="cpu",
        generator=seed_g,
    )
    negtive_text_embeddings = torch.randn(
        *text_shape,
        dtype=RUN_DTYPE,
        device="cpu",
        generator=seed_g,
    )

    if kwargs.get("attn_backend", None) is not None:
        backend = kwargs["attn_backend"]
        if backend == KsanaAttentionBackend.RADIAL_SAGE_ATTN:
            attention_config = KsanaRadialSageAttentionConfig(
                dense_attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN)
            )
        else:
            attention_config = KsanaAttentionConfig(backend=backend)
    else:
        attention_config = None

    if kwargs.get("lora_config", None) is not None:
        lora = nodes.build_list_of_lora_config(kwargs["lora_config"])
    else:
        lora = None

    load_output = nodes.KsanaNodeModelLoader.load(
        high_noise_model_path=model_path,
        low_noise_model_path=kwargs.get("low_noise_model_path", None),
        attention_config=attention_config,
        linear_backend=kwargs.get("linear_backend", KsanaLinearBackend.DEFAULT),
        model_boundary=kwargs.get("model_boundary", None),
        lora=lora,
        torch_compile_args=kwargs.get("torch_compile_args", None),
    )

    image_latent = torch.zeros(*image_latent_shape, dtype=RUN_DTYPE, device="cpu")
    batch_size_per_prompts = kwargs.get("batch_size_per_prompts", 1)
    generate_output = nodes.generate(
        load_output,
        positive=[[positive_text_embeddings]],
        negative=[[negtive_text_embeddings]],
        latent_image=nodes.KsanaNodeVAEEncodeOutput(
            samples=image_latent, batch_size_per_prompts=batch_size_per_prompts
        ),
        steps=steps,
        seed=SEED,
        cache_config=kwargs.get("cache_config", None),
        rope_function=kwargs.get("rope_function", "default"),
        low_sample_guide_scale=kwargs.get("low_sample_guide_scale", None),
    )
    return load_output, generate_output
