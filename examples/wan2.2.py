import os
import torch
import argparse

# set env before import ksana
os.environ["KSANA_LOGGER_LEVEL"] = "INFO"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from ksana import KsanaGenerator
from ksana.config import (
    KsanaModelConfig,
    KsanaTorchCompileConfig,
    KsanaDistributedConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)

# TODO: add more prompts to test bs > 1
prompt = (
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"
)
SEED = 1234
num_gpus = int(os.getenv("WORLD_SIZE", "1"))


def run_simple(model_dir):
    generator = KsanaGenerator.from_models(f"{model_dir}/Wan2.2-T2V-A14B", num_gpus=num_gpus)

    video = generator.generate_video(
        prompt,
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(720, 480),
            frame_num=17,
            cache_method="DCache",
            return_frames=True,
        ),
    )
    print("video shape:", video.shape)


def run_fp8_simple(model_dir):
    low_noise_model_path = (
        "/group/40164/ai-draw/models_prod/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    )
    high_noise_model_path = (
        "/group/40164/ai-draw/models_prod/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )
    text_dir = f"{model_dir}/Wan2.2-T2V-A14B"
    vae_dir = f"{model_dir}/Wan2.2-T2V-A14B"
    generator = KsanaGenerator.from_models(
        (low_noise_model_path, high_noise_model_path),
        text_checkpoint_dir=text_dir,
        vae_checkpoint_dir=vae_dir,
        num_gpus=num_gpus,  # high first
    )

    video = generator.generate_video(
        prompt,
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(720, 480),
            frame_num=17,
            cache_method="DCache",
            return_frames=True,
        ),
    )
    print("video shape:", video.shape)


def run_with_lora(model_dir):
    generator = KsanaGenerator.from_models(
        f"{model_dir}/Wan2.2-T2V-A14B",
        lora_dir=f"{model_dir}/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
    )

    generator.generate_video(prompt, runtime_config=KsanaRuntimeConfig(seed=SEED))


def run_advanced(model_dir):
    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attn_backend="flash_attention",
        # linear_backend="fp8_gemm",
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    generator = KsanaGenerator.from_models(
        f"{model_dir}/Wan2.2-T2V-A14B",
        num_gpus=num_gpus,
        model_config=model_config,
        dist_config=KsanaDistributedConfig(),
    )

    runtime_config = KsanaRuntimeConfig(
        size=(720, 480),
        seed=SEED,
        frame_num=17,
        cache_method="DCache",
        return_frames=True,
        output_folder="my_videos",
        save_video=True,
    )
    sample_config = KsanaSampleConfig(steps=40, cfg_scale=3.0, shift=12.0, solver="uni_pc")

    # Generate the video
    video = generator.generate_video(
        prompt,
        sample_config=sample_config,
        runtime_config=runtime_config,
    )
    print("video shape:", video.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 视频生成示例")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./",
        help="模型目录路径",
    )

    args = parser.parse_args()

    run_simple(args.model_dir)
    # run_fp8_simple(args.model_dir)
    run_with_lora(args.model_dir)
    run_advanced(args.model_dir)
