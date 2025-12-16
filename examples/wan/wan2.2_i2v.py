import os
import argparse

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaGenerator
from ksana.config import (
    KsanaDistributedConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
)


prompts = [
    "女孩扇子轻微挥动,吹口仙气后,手上的闪电飞到空中开始打雷",
    "新中式，戴发簪的女子，改良汉服（半透明丝绸），竹林，雾气，空灵氛围，丁达尔效应，清冷优雅，超写实",
    "缓慢的平移镜头，在外滩边上，有清风吹过。镜头从远到近，女孩在手舞足蹈的跳舞，舞姿非常美丽，镜头从远景到近景，给出了女孩的特写和细节。",
]

SEED = 1234


# def run_simple(args):
#     generator = KsanaGenerator.from_models(
#         f"{args.model_dir}/Wan2.2-I2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus)
#     )

#     video = generator.generate_video(
#         prompts,
#         img_path=args.img_path,
#         sample_config=KsanaSampleConfig(steps=40),
#         runtime_config=KsanaRuntimeConfig(
#             seed=SEED,
#             size=(1280, 720),
#             frame_num=81,
#             # cache_method="DCache",
#             return_frames=True,
#         ),
#     )

#     video = generator.generate_video(
#         prompts[0],
#         img_path=args.img_path,
#         sample_config=KsanaSampleConfig(steps=40),
#         runtime_config=KsanaRuntimeConfig(
#             seed=SEED,
#             size=(1280, 720),
#             frame_num=81,
#             # cache_method="DCache",
#             return_frames=True,
#         ),
#     )
#     print("video shape:", video.shape)


def run_start_and_end(args):
    generator = KsanaGenerator.from_models(
        f"{args.model_dir}/Wan2.2-I2V-A14B", dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus)
    )

    video = generator.generate_video(
        prompts[2],
        img_path="./examples/images/start_image.png",
        end_img_path="./examples/images/end_image.png",
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=(1280, 720),
            frame_num=81,
            return_frames=True,
        ),
    )

    print("video shape:", video.shape)


if __name__ == "__main__":
    """examples:
    - single card run:
        python examples/wan/wan2.2_i2v.py
    - run with multi-gpus has two ways:
        - CUDA_VISIBLE_DEVICES=4,5 python examples/wan/wan2.2_i2v.py --num_gpus=2
        - CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 examples/wan/wan2.2_i2v.py --num_gpus=2
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
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="使用的 GPU 数量",
    )

    args = parser.parse_args()

    # run_simple(args)
    run_start_and_end(args)
