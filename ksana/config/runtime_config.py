from dataclasses import dataclass, field

from ..utils.const import (
    DEFAULT_BATCHSIZE_PER_PROMPT,
    DEFAULT_OFFLOAD_MODEL,
    DEFAULT_OUTPUTS_VIDEO_DIR,
    DEFAULT_RETURN_FRAMES,
    DEFAULT_ROPE_FUNC_TYPE,
    DEFAULT_SAVE_OUTPUT,
)


@dataclass(frozen=True)
class KsanaRuntimeConfig:
    """_summary_
    size: tuple[int, int] = field(default=None): target image or video image size
    """

    size: tuple[int, int] | None = field(default=None, metadata={"help": "width and height of target size"})
    frame_num: int | None = field(default=None, metadata={"help": "number of frames to generate"})
    batch_size_per_prompts: int | list[int] | None = field(
        default=DEFAULT_BATCHSIZE_PER_PROMPT, metadata={"help": "batch size for every prompt"}
    )
    seed: int | None = field(default=None)
    offload_model: bool | None = field(default=DEFAULT_OFFLOAD_MODEL)
    rope_function: str | None = field(default=DEFAULT_ROPE_FUNC_TYPE)

    return_frames: bool | None = field(default=DEFAULT_RETURN_FRAMES)
    output_folder: str | None = field(default=DEFAULT_OUTPUTS_VIDEO_DIR)
    save_output: bool | None = field(default=DEFAULT_SAVE_OUTPUT)

    def __post_init__(self):
        if self.rope_function not in ["comfy", "default"]:
            raise ValueError("rope_function must be either 'comfy' or 'default'")
