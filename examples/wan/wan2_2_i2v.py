import argparse
import os

import torch

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaPipeline
from ksana.config import (
    KsanaAttentionBackend,
    KsanaAttentionConfig,
    KsanaDistributedConfig,
    KsanaLoraConfig,
    KsanaModelConfig,
    KsanaRuntimeConfig,
    KsanaSageSLAConfig,
    KsanaSampleConfig,
    KsanaSolverType,
)
from ksana.utils.distribute import get_gpu_count

prompts = [
    "女孩扇子轻微挥动,吹口仙气后,手上的闪电飞到空中开始打雷",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
    "缓慢的平移镜头，在外滩边上，有清风吹过。镜头从远到近，女孩在动作的特写，舞姿非常美丽，舞姿缓慢的移动，最后定格。镜头从远景到近景，外滩的背景清晰，给出了女孩的特写和细节",
]

SEED = 1234
NUM_GPUS = get_gpu_count()


def run_simple(args):
    pipeline = KsanaPipeline.from_models(
        f"{args.model_dir}/Wan2.2-I2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=NUM_GPUS)
    )

    video = pipeline.generate(
        prompts,
        img_path=args.img_path,
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(512, 512),
            frame_num=17,
            return_frames=True,
        ),
    )

    video = pipeline.generate(
        prompts[0],
        img_path=args.img_path,
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(1280, 720),
            frame_num=81,
            return_frames=True,
        ),
    )
    print("video shape:", video.shape)


def run_start_and_end_with_lora(args):

    model_config = KsanaModelConfig(
        run_dtype=torch.float16,
        attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN),
    )

    pipeline = KsanaPipeline.from_models(
        f"{args.model_dir}/Wan2.2-I2V-A14B",
        dist_config=KsanaDistributedConfig(num_gpus=NUM_GPUS),
        model_config=model_config,
        lora_config=KsanaLoraConfig(f"{args.model_dir}/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"),
    )

    sample_config = KsanaSampleConfig(
        steps=4,
        cfg_scale=1.0,
        shift=1.0,
        solver=KsanaSolverType.EULER,
        sigmas=[1.0, 0.9375001, 0.8333333, 0.625, 0.0000],
    )

    video = pipeline.generate(
        prompts[2],
        img_path="./examples/images/start_image.png",
        end_img_path="./examples/images/end_image.png",
        sample_config=sample_config,
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(512, 512),
            frame_num=49,
            return_frames=True,
        ),
    )

    print("video shape:", video.shape)


def run_turbo_diffusion(args):
    sage_sla_config = KsanaSageSLAConfig(
        dense_attention_config=KsanaAttentionConfig(backend=KsanaAttentionBackend.SAGE_ATTN), topk=0.1
    )

    model_config = KsanaModelConfig(attention_config=sage_sla_config, run_dtype=torch.bfloat16)

    sample_config = KsanaSampleConfig(steps=4, cfg_scale=1.0, shift=5.0, solver=KsanaSolverType.EULER)

    high = f"{args.model_dir}/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-high-720P.pth"
    low = f"{args.model_dir}/TurboWan2.2-I2V-A14B-720P/TurboWan2.2-I2V-A14B-low-720P.pth"
    text_dir = f"{args.model_dir}/Wan2.2-I2V-A14B"
    vae_dir = f"{args.model_dir}/Wan2.2-I2V-A14B"

    pipeline = KsanaPipeline.from_models(
        (high, low),
        text_checkpoint_dir=text_dir,
        vae_checkpoint_dir=vae_dir,
        dist_config=KsanaDistributedConfig(num_gpus=NUM_GPUS),
        model_config=model_config,
    )

    text = "POV selfie video, ultra-messy and extremely fast. A white cat in sunglasses "
    "stands on a surfboard with a neutral look when the board suddenly whips sideways,"
    "throwing cat and camera into the water; the frame dives sharply downward, swallowed "
    "by violent bursts of bubbles, spinning turbulence, and smeared water streaks as the "
    "camera sinks. Shadows thicken, pressure ripples distort the edges, and loose bubbles "
    "rush upward past the lens, showing the camera is still sinking. Then the cat kicks "
    "upward with explosive speed, dragging the view through churning bubbles and rapidly "
    "brightening water as sunlight floods back in; the camera races upward, water streaming "
    "off the lens, and finally breaks the surface in a sudden blast of light and spray, snapping "
    "back into a crooked, frantic selfie as the cat resurfaces."

    pipeline.generate(
        text,
        img_path="./examples/images/cat.png",
        runtime_config=KsanaRuntimeConfig(size=(1280, 720), seed=SEED, frame_num=81),
        sample_config=sample_config,
    )


if __name__ == "__main__":
    """examples:
    - single card run:
        python examples/wan/wan2_2_i2v.py
    - run with multi-gpus has two ways:
        - CUDA_VISIBLE_DEVICES=4,5 python examples/wan/wan2_2_i2v.py
        - CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 examples/wan/wan2_2_i2v.py
    """
    parser = argparse.ArgumentParser(description="Wan2.2 视频生成示例")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./",
        help="模型目录路径",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="./examples/images/input.png",
        help="输入图片路径",
    )

    args = parser.parse_args()

    run_simple(args)
    run_start_and_end_with_lora(args)
    run_turbo_diffusion(args)
