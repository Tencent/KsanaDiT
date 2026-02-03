import binascii
import os
import shutil
import subprocess

import imageio
import torch
import torchvision
from PIL import Image

from .logger import log


def merge_video_audio(video_path: str, audio_path: str):
    """
    Merge the video and audio into a new video, with the duration set to the shorter of the two,
    and overwrite the original video file.

    Parameters:
    video_path (str): Path to the original video file
    audio_path (str): Path to the audio file
    """
    # check
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video file {video_path} does not exist")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"audio file {audio_path} does not exist")

    base, ext = os.path.splitext(video_path)
    temp_output = f"{base}_temp{ext}"

    try:
        # create ffmpeg command
        command = [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",  # copy video stream
            "-c:a",
            "aac",  # use AAC audio encoder
            "-b:a",
            "192k",  # set audio bitrate (optional)
            "-map",
            "0:v:0",  # select the first video stream
            "-map",
            "1:a:0",  # select the first audio stream
            "-shortest",  # choose the shortest duration
            temp_output,
        ]

        # execute the command
        log.info("Start merging video and audio...")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # check result
        if result.returncode != 0:
            error_msg = f"FFmpeg execute failed: {result.stderr}"
            log.error(error_msg)
            raise RuntimeError(error_msg)

        shutil.move(temp_output, video_path)
        log.info(f"Merge completed, saved to {video_path}")

    except Exception as e:  # pylint: disable=broad-except
        if os.path.exists(temp_output):
            os.remove(temp_output)
        log.error(f"merge_video_audio failed with error: {e}")


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def save_video(tensor, save_file=None, fps=30, suffix=".mp4", nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    cache_file = os.path.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack(
            [
                torchvision.utils.make_grid(u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
            dim=1,
        ).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:  # pylint: disable=broad-except
        log.info(f"save_video failed, error: {e}")


def load_video_frames(video_path: str, max_frames: int = 81) -> torch.Tensor:
    """
    Load video frames from a video file.

    Args:
        video_path: Path to video file (mp4, etc.)
        max_frames: Maximum number of frames to load

    Returns:
        Tensor of shape [N, H, W, C] in range [0, 1]
    """
    import torchvision.io as io

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist")

    # Read video frames
    video, _, _ = io.read_video(video_path, pts_unit="sec")
    # video shape: [T, H, W, C] in [0, 255]

    # Limit frames
    if video.shape[0] > max_frames:
        video = video[:max_frames]

    # Normalize to [0, 1]
    video = video.float() / 255.0

    log.info(f"Loaded video from {video_path}: shape={video.shape}")
    return video


def save_image(tensor: torch.Tensor, path: str):
    """Save tensor as image file."""
    img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().float().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path)
    print(f"Saved image to {path}")


def save_images(images: torch.Tensor, save_paths: list[str]):
    if images.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {images.dim()}D tensor")

    batch_size = images.shape[0]
    if len(save_paths) != batch_size:
        raise ValueError(f"Number of save paths ({len(save_paths)}) must match batch size ({batch_size})")

    for i, save_path in enumerate(save_paths):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(images[i : i + 1], save_path)


def load_control_frames(control_path: str, max_frames: int, target_size: tuple[int, int] | None = None) -> torch.Tensor:
    import numpy as np

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    _, ext = os.path.splitext(control_path.lower())

    def _image_to_tensor(img: Image.Image) -> torch.Tensor:
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(img)).float()
        return tensor.unsqueeze(0) / 255.0

    if ext in image_exts:
        image = Image.open(control_path).convert("RGB")
        return _image_to_tensor(image)

    try:
        video = load_video_frames(control_path, max_frames=max_frames)
        # video shape: [N, H, W, C] in [0, 1]
        if target_size is not None:
            # Resize each frame
            frames = []
            for i in range(video.shape[0]):
                frame_np = (video[i].numpy() * 255).astype(np.uint8)
                pil_frame = Image.fromarray(frame_np).resize(target_size, Image.Resampling.LANCZOS)
                frames.append(torch.from_numpy(np.array(pil_frame)).float() / 255.0)
            video = torch.stack(frames, dim=0)
        return video
    except (RuntimeError, OSError, ValueError) as e:
        # Fallback to image loading if video reader fails.
        log.info(f"Video loading failed, falling back to image: {e}")
        image = Image.open(control_path).convert("RGB")
        return _image_to_tensor(image)


def match_control_frames(control_video: torch.Tensor, target_frames: int) -> torch.Tensor:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")

    current_frames = control_video.shape[0]
    if current_frames == target_frames:
        return control_video
    if current_frames > target_frames:
        return control_video[:target_frames]

    # Repeat last frame to match requested length (common for single image control).
    repeat_count = target_frames - current_frames
    last_frame = control_video[-1:].repeat(repeat_count, 1, 1, 1)
    return torch.cat([control_video, last_frame], dim=0)
