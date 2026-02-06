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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..utils.const import DEFAULT_DENOISE
from .video_control_config import KsanaVideoControlConfig


class KsanaSolverType(Enum):
    UNI_PC = "uni_pc"
    EULER = "euler"
    FLOWMATCH_EULER = "flowmatch_euler"
    # DPM_PLUS_PLUS = "dpm++" #TODO: 这个还需要check，里面有两次 shift 操作，是否合理
    DPM_PLUS_PLUS_SDE = "dpm++_sde"

    @staticmethod
    def get_supported_list() -> list[str]:
        return [b.value for b in KsanaSolverType]

    @staticmethod
    def support(type: str) -> bool:
        if isinstance(type, str):
            return type in KsanaSolverType.get_supported_list()
        elif isinstance(type, KsanaSolverType):
            return True
        else:
            return False


@dataclass(frozen=True)
class KsanaSampleConfig:
    steps: int | None = field(default=None)
    cfg_scale: float | list[float, float] | None = field(default=None)
    shift: float | None = field(default=None)
    solver: KsanaSolverType | None = field(default=None)
    denoise: float | None = field(default=DEFAULT_DENOISE)
    sigmas: list[float] | None = field(default=None)
    video_control: KsanaVideoControlConfig | None = field(default=None)
    add_noise_to_latent: bool = field(default=False)  # Whether to add noise to the input latent

    def __post_init__(self):
        if self.solver is not None and not isinstance(self.solver, KsanaSolverType):
            raise ValueError(f"solver must be a KsanaSolverType enum, got {type(self.solver)}")
