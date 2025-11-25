import torch

from dataclasses import dataclass, field
from easydict import EasyDict

from ..utils.const import DEFAULT_OFFLOAD_MODEL, DEFAULT_RETURN_FRAMES, DEFAULT_OUTPUTS_VIDEO_DIR, DEFAULT_SAVE_VIDEO

from ..cache import SUPPORTED_CACHE_METHODS


@dataclass(frozen=True)
class KsanaRuntimeConfig:
    """_summary_
    size: tuple[int, int] = field(default=None): target image or video image size
    """

    size: tuple[int, int] | None = field(default=None)
    frame_num: int | None = field(default=None)
    seed: int | None = field(default=None)
    run_dtype: torch.dtype | None = field(default=None)
    offload_model: bool | None = field(default=None)
    boundary: float | None = field(default=None)

    return_frames: bool | None = field(default=None)
    output_folder: str | None = field(default=None)
    save_video: bool | None = field(default=None)

    cache_method: str | None = field(default=None)

    def __post_init__(self):
        assert (
            self.cache_method in SUPPORTED_CACHE_METHODS
        ), f"unsupported cache method {self.cache_method}, not in {SUPPORTED_CACHE_METHODS}"

    @staticmethod
    def copy_with_default(input_config, default: dict | EasyDict):
        return KsanaRuntimeConfig(
            size=default.get("size", None) if input_config.size is None else input_config.size,
            frame_num=default.get("frame_num", None) if input_config.frame_num is None else input_config.frame_num,
            seed=input_config.seed,
            run_dtype=default.get("param_dtype", None) if input_config.run_dtype is None else input_config.run_dtype,
            offload_model=DEFAULT_OFFLOAD_MODEL if input_config.offload_model is None else input_config.offload_model,
            boundary=default.get("boundary", None) if input_config.boundary is None else input_config.boundary,
            return_frames=DEFAULT_RETURN_FRAMES if input_config.return_frames is None else input_config.return_frames,
            output_folder=(
                DEFAULT_OUTPUTS_VIDEO_DIR if input_config.output_folder is None else input_config.output_folder
            ),
            save_video=DEFAULT_SAVE_VIDEO if input_config.save_video is None else input_config.save_video,
            cache_method=input_config.cache_method,
        )
