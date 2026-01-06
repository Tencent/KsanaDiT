import binascii
import os
import shutil
import subprocess

import imageio
import torch
import torchvision

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

    except Exception as e:
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
    except Exception as e:
        log.info(f"save_video failed, error: {e}")
