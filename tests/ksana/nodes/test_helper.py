import torch

TARGET_T2V_IMG_SHAPE = [1, 16, 16, 32, 32]
TARGET_I2V_IMG_SHAPE = [1, 20, 16, 32, 32]
TARGET_I2I_IMG_SHAPE = [1, 16, 1, 32, 32]

TEST_STEPS = 2

WAN_TEXT_SHAPE = [1, 512, 4096]
QWEN_TEXT_SHAPE = [1, 1024, 3584]

COMFY_MODEL_ROOT = "/data/stable-diffusion-webui/models"

SEED = 321
RUN_DTYPE = torch.float16

TEST_MODELS = [
    ("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors", TARGET_T2V_IMG_SHAPE, WAN_TEXT_SHAPE),
    ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", TARGET_I2V_IMG_SHAPE, WAN_TEXT_SHAPE),
    ("wan2.2_t2v_high_noise_14B_fp16.safetensors", TARGET_T2V_IMG_SHAPE, WAN_TEXT_SHAPE),
    ("wan2.2_i2v_high_noise_14B_fp16.safetensors", TARGET_I2V_IMG_SHAPE, WAN_TEXT_SHAPE),
    ("qwen_image_2512_fp8_e4m3fn.safetensors", TARGET_I2I_IMG_SHAPE, QWEN_TEXT_SHAPE),
]

TEST_ONE_GPU_EPS_PLACE = 5
TEST_GPUS_EPS_PLACE = 5
