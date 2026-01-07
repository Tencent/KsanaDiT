import argparse
import os

import torch

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaGenerator
from ksana.config import (
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaTorchCompileConfig,
)
from ksana.config.cache_config import CustomStepCacheConfig, DBCacheConfig, DCacheConfig, KsanaHybridCacheConfig
from ksana.operations import KsanaAttentionBackend, KsanaLinearBackend

prompts = [
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
]

SEED = 1234


def run_simple(args):
    generator = KsanaGenerator.from_models(
        f"{args.model_dir}/Wan2.2-T2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus)
    )

    video = generator.generate(
        prompts[0],
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(720, 480),
            frame_num=17,
            return_frames=True,
        ),
        cache_config=DCacheConfig(),
    )
    print("video shape:", video.shape)


def run_fp8_models(args):
    low_noise_model_path = (
        f"{args.model_dir}/comfy_models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
    )
    high_noise_model_path = (
        f"{args.model_dir}/comfy_models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )
    text_dir = f"{args.model_dir}/Wan2.2-T2V-A14B"
    vae_dir = f"{args.model_dir}/Wan2.2-T2V-A14B"

    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
        linear_backend=KsanaLinearBackend.FP8_GEMM,
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    generator = KsanaGenerator.from_models(
        (high_noise_model_path, low_noise_model_path),  # high go first
        text_checkpoint_dir=text_dir,
        vae_checkpoint_dir=vae_dir,
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
        model_config=model_config,
    )
    high_cache_config = DCacheConfig(fast_degree=55)
    low_cache_config = DCacheConfig(fast_degree=45)

    video = generator.generate(
        prompts[0],
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(720, 480),
            frame_num=17,
            return_frames=True,
        ),
        cache_config=[high_cache_config, low_cache_config],
    )
    print("video shape:", video.shape)


def run_with_lora(args):
    generator = KsanaGenerator.from_models(
        f"{args.model_dir}/Wan2.2-T2V-A14B",
        lora=f"{args.model_dir}/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    generator.generate(prompts, runtime_config=KsanaRuntimeConfig(seed=SEED), cache_config=DCacheConfig())


def run_advanced(args):
    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.FLASH_ATTN),
        torch_compile_config=KsanaTorchCompileConfig(),
    )
    generator = KsanaGenerator.from_models(
        f"{args.model_dir}/Wan2.2-T2V-A14B",
        model_config=model_config,
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    runtime_config = KsanaRuntimeConfig(
        size=(720, 480),
        seed=SEED,
        frame_num=17,
        return_frames=True,
        output_folder="outputs",
        save_video=True,
    )

    sample_config = KsanaSampleConfig(steps=40, cfg_scale=3.0, shift=12.0, solver="uni_pc")

    cache_config = KsanaHybridCacheConfig(
        step_cache=DCacheConfig(fast_degree=50),
        block_cache=DBCacheConfig(),
    )

    # Generate the video
    video = generator.generate(
        prompts[0], sample_config=sample_config, runtime_config=runtime_config, cache_config=cache_config
    )
    print("video shape:", video.shape)


def run_fast(args):
    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.FLASH_ATTN),
        torch_compile_config=KsanaTorchCompileConfig(mode="max-autotune-no-cudagraphs"),
    )
    generator = KsanaGenerator.from_models(
        f"{args.model_dir}/Wan2.2-T2V-A14B",
        model_config=model_config,
        lora=f"{args.model_dir}/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    runtime_config = KsanaRuntimeConfig(
        size=(1280, 720),
        seed=SEED,
        frame_num=81,
        return_frames=True,
        output_folder="outputs",
        save_video=True,
        rope_function="comfy",
        boundary=0.9,
    )

    sample_config = KsanaSampleConfig(
        steps=4, cfg_scale=1.0, shift=5.0, solver="euler", sigmas=[1.0, 0.9375001, 0.6333333, 0.225, 0.0000]
    )

    cache_config = CustomStepCacheConfig(steps=3, scales=1.1)

    # Generate the video
    video = generator.generate(
        prompts[0], sample_config=sample_config, runtime_config=runtime_config, cache_config=cache_config
    )
    print("video shape:", video.shape)


if __name__ == "__main__":
    """examples:
    - single card run:
        python examples/wan/wan2.2_t2v.py
    - run with multi-gpus has two ways:
        - CUDA_VISIBLE_DEVICES=4,5 python examples/wan/wan2.2_t2v.py --num_gpus=2
        - CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 examples/wan/wan2.2_t2v.py --num_gpus=2
    """
    parser = argparse.ArgumentParser(description="Wan2.2 视频生成示例")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./",
        help="模型目录路径",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="使用的 GPU 数量",
    )

    args = parser.parse_args()

    run_simple(args)
    run_fp8_models(args)
    run_with_lora(args)
    run_advanced(args)
    run_fast(args)
