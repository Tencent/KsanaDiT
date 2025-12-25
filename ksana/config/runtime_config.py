from dataclasses import dataclass, field
from easydict import EasyDict

from ..utils.const import DEFAULT_OFFLOAD_MODEL, DEFAULT_RETURN_FRAMES, DEFAULT_OUTPUTS_VIDEO_DIR, DEFAULT_SAVE_VIDEO

from ..cache import SUPPORTED_CACHE_METHODS


@dataclass(frozen=True)
class KsanaRuntimeConfig:
    """_summary_
    size: tuple[int, int] = field(default=None): target image or video image size
    """

    size: tuple[int, int] | None = field(default=None, metadata={"help": "width and height of target size"})
    frame_num: int | None = field(default=None, metadata={"help": "number of frames to generate"})
    seed: int | None = field(default=None)
    offload_model: bool | None = field(default=DEFAULT_OFFLOAD_MODEL)
    boundary: float | None = field(default=None)
    rope_function: str | None = field(default="default")

    return_frames: bool | None = field(default=DEFAULT_RETURN_FRAMES)
    output_folder: str | None = field(default=DEFAULT_OUTPUTS_VIDEO_DIR)
    save_video: bool | None = field(default=DEFAULT_SAVE_VIDEO)

    cache_method: str | None = field(default=None)

    def __post_init__(self):
        assert (
            self.cache_method is None or self.cache_method in SUPPORTED_CACHE_METHODS
        ), f"unsupported cache method {self.cache_method}, not in {SUPPORTED_CACHE_METHODS}"

    @staticmethod
    def copy_with_default(input_config, default: dict | EasyDict):
        return KsanaRuntimeConfig(
            size=default.get("size", None) if input_config.size is None else input_config.size,
            frame_num=default.get("frame_num", None) if input_config.frame_num is None else input_config.frame_num,
            seed=input_config.seed,
            offload_model=input_config.offload_model,
            boundary=default.get("boundary", None) if input_config.boundary is None else input_config.boundary,
            rope_function=input_config.rope_function,
            return_frames=input_config.return_frames,
            output_folder=input_config.output_folder,
            save_video=input_config.save_video,
            cache_method=input_config.cache_method,
        )
