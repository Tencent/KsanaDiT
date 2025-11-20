import os

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from ksana import KsanaGenerator, KsanaTorchCompileConfig

prompt = (
    "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"
)
seed = 1234


def run_simple():
    generator = KsanaGenerator.from_pretrained(
        "./Wan2.2-T2V-A14B",
        torch_compile_config=KsanaTorchCompileConfig(),
    )

    # Generate the video
    video = generator.generate_video(
        prompt,
        steps=40,
        size=(720, 480),
        seed=seed,
        frame_num=17,  # 81
        cache_method="DCache",
        return_frames=True,
        output_folder="my_videos",
    )
    print("video shape:", video.shape)


def run_with_lora():
    generator = KsanaGenerator.from_pretrained(
        "./Wan2.2-T2V-A14B",
        lora_dir="./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1",
    )

    # Generate the video
    video = generator.generate_video(
        prompt,
        seed=seed,
        return_frames=True,
        output_folder="my_videos",
    )
    print("video shape:", video.shape)


if __name__ == "__main__":
    run_simple()
    run_with_lora()
