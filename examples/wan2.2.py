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


prompts = [
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 1234
num_gpus = int(os.getenv("WORLD_SIZE", "1"))


def run_simple(model_dir):
    generator = KsanaGenerator.from_models(f"{model_dir}/Wan2.2-T2V-A14B", num_gpus=num_gpus)

    video = generator.generate_video(
        prompts[0],
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


def run_fp8_models(model_dir):
    low_noise_model_path = f"{model_dir}/comfy_models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    high_noise_model_path = (
        f"{model_dir}/comfy_models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )
    text_dir = f"{model_dir}/Wan2.2-T2V-A14B"
    vae_dir = f"{model_dir}/Wan2.2-T2V-A14B"

    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attn_backend="sage_attention",
        linear_backend="fp8_gemm",
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    generator = KsanaGenerator.from_models(
        (high_noise_model_path, low_noise_model_path),  # high go first
        text_checkpoint_dir=text_dir,
        vae_checkpoint_dir=vae_dir,
        num_gpus=num_gpus,
        model_config=model_config,
    )

    video = generator.generate_video(
        prompts[0],
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

    generator.generate_video(prompts, runtime_config=KsanaRuntimeConfig(seed=SEED))


def run_with_lora_in_distributed_mode(model_dir):
    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attn_backend="flash_attention",
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    dist_config = KsanaDistributedConfig(
        world_size=num_gpus,
        use_sp=True,
        dit_fsdp=False,
        ulysses_size=num_gpus,
    )

    generator = KsanaGenerator.from_models(
        f"{model_dir}/Wan2.2-T2V-A14B",
        lora_dir=f"{model_dir}/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
        model_config=model_config,
        num_gpus=num_gpus,
        dist_config=dist_config,
    )

    generator.generate_video(
        prompts[0],
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            output_folder="distributed_videos",
            save_video=True,
        ),
    )


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
        output_folder="outputs",
        save_video=True,
    )
    sample_config = KsanaSampleConfig(steps=40, cfg_scale=3.0, shift=12.0, solver="uni_pc")

    # Generate the video
    video = generator.generate_video(
        prompts[0],
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
    run_fp8_models(args.model_dir)
    run_with_lora(args.model_dir)
    run_advanced(args.model_dir)
    # run with torchrun: torchrun --nproc_per_node=2 examples/wan2.2.py
    # run_with_lora_in_distributed_mode(args.model_dir)
