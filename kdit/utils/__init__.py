# Copyright 2025 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..accelerator.dtype import normalize_dtype_for_platform
from .debug import print_recursive
from .device import get_intermediate_device
from .distribute import all_to_all, gather_forward, get_gpu_count, get_rank_id, get_world_size
from .env import KSANA_LOGGER_LEVEL, KSANA_MEMORY_PROFILER
from .experimental_sampling import (
    ExperimentalSamplingUtils,
    compute_cfg_zero_star_alpha,
    compute_raag_guidance,
    fourier_filter,
    tangential_projection,
    temporal_score_rescaling,
)
from .factory import Factory
from .instance import singleton
from .load import load_file_to_state_dict, load_sharded_safetensors, remove_prefix_from_state_dict
from .logger import log, reset_logging
from .lora import load_state_dict_and_merge_lora, model_safe_downcast
from .media import load_control_frames, load_video_frames, match_control_frames, merge_video_audio, save_video
from .monitor import report, report_inner
from .ops import cast_bias_weight, common_upscale, stochastic_rounding, supports_fp8_compute
from .profile import KsanaProfiler, MemoryProfiler, nvtx_range, time_range
from .sample_solver import apply_sigma_shift, get_sigmas_with_denoise
from .types import any_key_in_str, evolve_with_recommend, is_file_or_dir, str_to_list
from .vace import (
    KsanaVaceContext,
    apply_bidirectional_sampling,
    apply_experimental_cfg,
    apply_temporal_score_rescaling,
    apply_vace_trim,
    build_vace_kwargs,
    get_step_video_control,
    init_latent_stats,
    latent_process_in,
    latent_process_out,
    parse_video_control_kwargs,
)

__all__ = [
    "Factory",
    "report_inner",
    "report",
    "KSANA_LOGGER_LEVEL",
    "KSANA_MEMORY_PROFILER",
    "KsanaProfiler",
    "MemoryProfiler",
    "nvtx_range",
    "log",
    "reset_logging",
    "singleton",
    "normalize_dtype_for_platform",
    "evolve_with_recommend",
    "str_to_list",
    "is_file_or_dir",
    "any_key_in_str",
    "print_recursive",
    "time_range",
    "save_video",
    "merge_video_audio",
    "load_video_frames",
    "load_control_frames",
    "match_control_frames",
    "get_rank_id",
    "get_world_size",
    "get_gpu_count",
    "gather_forward",
    "all_to_all",
    "load_state_dict_and_merge_lora",
    "model_safe_downcast",
    "load_file_to_state_dict",
    "load_sharded_safetensors",
    "cast_bias_weight",
    "common_upscale",
    "get_intermediate_device",
    "supports_fp8_compute",
    "stochastic_rounding",
    "get_sigmas_with_denoise",
    "apply_sigma_shift",
    "remove_prefix_from_state_dict",
    "ExperimentalSamplingUtils",
    "compute_cfg_zero_star_alpha",
    "tangential_projection",
    "compute_raag_guidance",
    "fourier_filter",
    "temporal_score_rescaling",
    "KsanaVaceContext",
    "apply_bidirectional_sampling",
    "apply_experimental_cfg",
    "apply_temporal_score_rescaling",
    "apply_vace_trim",
    "build_vace_kwargs",
    "get_step_video_control",
    "init_latent_stats",
    "latent_process_in",
    "latent_process_out",
    "parse_video_control_kwargs",
]
