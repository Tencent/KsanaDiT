import os

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

prompt = (
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"
)
seed = 1234
num_gpus = int(os.getenv("WORLD_SIZE", "1"))


def run_simple():
    generator = KsanaGenerator.from_pretrained("./Wan2.2-T2V-A14B", num_gpus=num_gpus)

    video = generator.generate_video(
        prompt,
        sample_config=KsanaSampleConfig(steps=40),
        runtime_config=KsanaRuntimeConfig(
            seed=seed, size=(720, 480), frame_num=17, cache_method="DCache", return_frames=True
        ),
    )
    if video is not None:
        print("video shape:", video.shape)


def run_with_lora():
    generator = KsanaGenerator.from_pretrained(
        "./Wan2.2-T2V-A14B",
        lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
    )

    # Generate the video
    generator.generate_video(prompt)


def run_advanced():
    model_config = KsanaModelConfig(
        weight_dtype="float16",
        attn_backend="flash_attention",
        linear_backend="fp8_gemm",
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    generator = KsanaGenerator.from_pretrained(
        "./Wan2.2-T2V-A14B",
        num_gpus=num_gpus,
        model_config=model_config,
        dist_config=KsanaDistributedConfig(),
    )

    runtime_config = KsanaRuntimeConfig(
        size=(720, 480),
        seed=seed,
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
    # run_simple()
    # run_with_lora()
    run_advanced()
