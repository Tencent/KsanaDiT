from dataclasses import dataclass, field
from easydict import EasyDict

from ..utils.const import (
    DEFAULT_OFFLOAD_MODEL,
    DEFAULT_RETURN_FRAMES,
    DEFAULT_OUTPUTS_VIDEO_DIR,
    DEFAULT_SAVE_VIDEO,
    DEFAULT_ROPE_FUNC_TYPE,
)


@dataclass(frozen=True)
class KsanaRuntimeConfig:
    """_summary_
    size: tuple[int, int] = field(default=None): target image or video image size
    """

    size: tuple[int, int] | None = field(default=None, metadata={"help": "width and height of target size"})
    frame_num: int | None = field(default=None, metadata={"help": "number of frames to generate"})
    batch_size_per_prompt: int | list[int] | None = field(default=1, metadata={"help": "batch size per prompt"})
    seed: int | None = field(default=None)
    offload_model: bool | None = field(default=DEFAULT_OFFLOAD_MODEL)
    boundary: float | None = field(default=None)
    rope_function: str | None = field(default=DEFAULT_ROPE_FUNC_TYPE)

    return_frames: bool | None = field(default=DEFAULT_RETURN_FRAMES)
    output_folder: str | None = field(default=DEFAULT_OUTPUTS_VIDEO_DIR)
    save_video: bool | None = field(default=DEFAULT_SAVE_VIDEO)

    @staticmethod
    def copy_with_default(input_config, default: dict | EasyDict):
        return KsanaRuntimeConfig(
            size=default.get("size", None) if input_config.size is None else input_config.size,
            frame_num=default.get("frame_num", None) if input_config.frame_num is None else input_config.frame_num,
            batch_size_per_prompt=input_config.batch_size_per_prompt,
            seed=input_config.seed,
            offload_model=input_config.offload_model,
            boundary=default.get("boundary", None) if input_config.boundary is None else input_config.boundary,
            rope_function=input_config.rope_function,
            return_frames=input_config.return_frames,
            output_folder=input_config.output_folder,
            save_video=input_config.save_video,
        )
