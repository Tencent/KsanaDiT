import argparse
import os

os.environ["KSANA_LOGGER_LEVEL"] = "INFO"

from ksana import KsanaPipeline
from ksana.config import (
    DCacheConfig,
    KsanaDistributedConfig,
    KsanaExperimentalConfig,
    KsanaFETAConfig,
    KsanaRuntimeConfig,
    KsanaSampleConfig,
    KsanaSLGConfig,
    KsanaSolverType,
    KsanaVideoControlConfig,
)
from ksana.utils import load_control_frames
from ksana.utils.vace import KsanaVaceVideoEncodeConfig

prompts = [
    "a cute anime girl with massive fennec ears and a big fluffy tail "
    "turning around and dancing and singing on stage like an idol",
]

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
SEED = 620012503742781


def run_simple(args):
    engine = KsanaPipeline.from_models(
        f"{args.model_dir}",
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    size = args.size if args.size else (512, 512)
    frame_num = args.num_frames

    video = engine.generate(
        prompts[0],
        prompt_negative=NEGATIVE_PROMPT,
        sample_config=KsanaSampleConfig(
            steps=args.steps,
            cfg_scale=6.0,
            shift=8.0,
            solver=KsanaSolverType.UNI_PC,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=size,
            frame_num=frame_num,
            return_frames=True,
        ),
        cache_config=[DCacheConfig()],
    )
    print(f"Generated video shape: {video.shape}")
    return video


def run_with_experimental_configs(args):
    engine = KsanaPipeline.from_models(
        f"{args.model_dir}",
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    # SLG Config: Skip uncond on blocks 9,10,11 for 14B model
    # This can speed up sampling by ~15-20% with minimal quality loss
    slg_config = KsanaSLGConfig(
        blocks=[9],  # Transformer block indices to skip uncond
        start_percent=0.1,  # Start SLG after 10% of steps
        end_percent=1.0,  # Enable until the end
    )

    # FETA Config: Enhance temporal consistency
    # Higher weight = stronger temporal smoothing
    feta_config = KsanaFETAConfig(
        weight=2.0,  # Enhancement weight (1.0-5.0 typical)
        start_percent=0.0,  # Start from beginning
        end_percent=1.0,  # Enable until the end
    )

    experimental_config = KsanaExperimentalConfig(
        cfg_zero_star=True,
        use_zero_init=False,
        zero_star_steps=0,
        # FreSca: Frequency-domain filtering
        use_fresca=False,
        fresca_scale_low=1.0,
        fresca_scale_high=1.25,
        fresca_freq_cutoff=20,
        # TCFG: Tangential CFG to reduce color shifts
        use_tcfg=False,
        # RAAG: Ratio-aware adaptive guidance (0 = disabled)
        raag_alpha=0.0,
        # TSR: Temporal score rescaling for consistency
        temporal_score_rescaling=False,
        tsr_k=0.95,
        tsr_sigma=1.0,
    )

    print(f"slg_config: {slg_config}")
    print(f"feta_config: {feta_config}")
    print(f"experimental_config: {experimental_config}")

    # Handle control video if provided
    video_control_config = None
    size = args.size if args.size else (512, 512)
    frame_num = args.num_frames

    if args.control_video:
        # Load as reference image (ComfyUI uses reference_image, not control_video)
        reference_image = load_control_frames(args.control_video, max_frames=1, target_size=size)
        print(f"Reference image shape: {reference_image.shape}")
        video_control_config = KsanaVaceVideoEncodeConfig(
            reference_image=reference_image,
            strength=args.vace_strength,
        )

    video = engine.generate(
        prompts[0],
        prompt_negative=NEGATIVE_PROMPT,
        sample_config=KsanaSampleConfig(
            steps=args.steps,
            cfg_scale=6.0,
            shift=8.0,
            solver=KsanaSolverType.UNI_PC,
            video_control=KsanaVideoControlConfig(
                slg=slg_config,
                feta=feta_config,
                experimental=experimental_config,
            ),
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=size,
            frame_num=frame_num,
            return_frames=True,
        ),
        cache_config=[DCacheConfig()],
        video_control_config=video_control_config,
    )
    print(f"Generated video shape: {video.shape}")
    return video


def run_with_control_video(args):
    engine = KsanaPipeline.from_models(
        f"{args.model_dir}",
        dist_config=KsanaDistributedConfig(num_gpus=args.num_gpus),
    )

    target_size = args.size if args.size else (512, 512)
    target_frames = args.num_frames

    # Load as reference image (ComfyUI uses reference_image, not control_video)
    reference_image = load_control_frames(args.control_video, max_frames=1, target_size=target_size)
    print(f"Reference image shape: {reference_image.shape}")

    video_control_config = KsanaVaceVideoEncodeConfig(
        reference_image=reference_image,
        strength=args.vace_strength,
    )

    video = engine.generate(
        prompts[0],
        prompt_negative=NEGATIVE_PROMPT,
        sample_config=KsanaSampleConfig(
            steps=args.steps,
            cfg_scale=6.0,
            shift=8.0,
            solver=KsanaSolverType.UNI_PC,
        ),
        runtime_config=KsanaRuntimeConfig(
            seed=SEED,
            size=target_size,
            frame_num=target_frames,
            return_frames=True,
        ),
        cache_config=[DCacheConfig()],
        video_control_config=video_control_config,
    )
    print(f"Generated video shape: {video.shape}")
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./",
        help="Path to the VACE model directory",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--control_video",
        type=str,
        default=None,
        help="Path to control video file (mp4, etc.)",
    )
    parser.add_argument(
        "--vace_strength",
        type=float,
        default=1.0,
        help="VACE conditioning strength",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Output video size as WIDTHxHEIGHT (e.g., 512x512). If not specified, defaults to 512x512.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--use_experimental",
        action="store_true",
        help="Enable experimental sampling optimizations (SLG, FETA, CFG-Zero-Star, etc.)",
    )

    args = parser.parse_args()

    # Parse size argument
    if args.size:
        try:
            w, h = args.size.lower().split("x")
            args.size = (int(w), int(h))
        except ValueError:
            parser.error(f"Invalid size format '{args.size}'. Use WIDTHxHEIGHT (e.g., 848x480)")

    if args.use_experimental:
        run_with_experimental_configs(args)
    elif args.control_video:
        run_with_control_video(args)
    else:
        run_simple(args)
