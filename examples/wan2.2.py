from ksana import KsanaGenerator, KsanaTorchCompileConfig
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    torch_compile_config = KsanaTorchCompileConfig()
    generator = KsanaGenerator.from_pretrained("./Wan2.2-T2V-A14B", torch_compile_config=torch_compile_config)

    # Define a prompt for your video
    prompt = "街头摄影，戴耳机的酷女孩滑板，纽约街头，涂鸦墙背景，动态姿势，风吹头发，黄金时刻光线，主体清晰背景虚化，街头潮牌。"

    # Generate the video
    video = generator.generate_video(
        prompt,
        steps=20,
        size=(720, 480),
        frame_num=17,
        cache_method="DCache",
        return_frames=True,
        output_folder="my_videos",
    )
    print(video.shape)


if __name__ == "__main__":
    main()
