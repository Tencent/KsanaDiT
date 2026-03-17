# Copyright 2026 Tencent
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

from .context_builder import ContextBuilder

# 导入 defs 子包触发自动注册（import 即注册到全局注册表）
from .defs import qwen_edit as _qwen_edit  # noqa: F401  # pylint: disable=unused-import
from .defs import qwen_t2i as _qwen_t2i  # noqa: F401  # pylint: disable=unused-import
from .defs import wan_i2v as _wan_i2v  # noqa: F401  # pylint: disable=unused-import
from .defs import wan_t2v as _wan_t2v  # noqa: F401  # pylint: disable=unused-import
from .defs import wan_vace as _wan_vace  # noqa: F401  # pylint: disable=unused-import
from .generate_inputs import GenerateInputs
from .pipeline import Pipeline
from .pipeline_def import (
    PipelineDef,
    PipelineDefBuilder,
    get_pipeline_def,
    register_pipeline_def,
)
from .pipeline_key import PipelineKey
from .pipeline_phase import InferPhase, LoadPhase

__all__ = [
    "Pipeline",
    "PipelineDef",
    "PipelineDefBuilder",
    "PipelineKey",
    "LoadPhase",
    "InferPhase",
    "ContextBuilder",
    "GenerateInputs",
    "register_pipeline_def",
    "get_pipeline_def",
]
