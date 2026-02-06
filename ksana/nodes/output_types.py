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

from dataclasses import dataclass, field

import torch

from ksana.models.model_key import KsanaModelKey


@dataclass
class KsanaNodeModelLoaderOutput:
    model: KsanaModelKey | list[KsanaModelKey] = field(default=None)
    model_name: str = field(default_factory=str)  # TODO(qian):  need remove
    run_dtype: torch.dtype | None = field(default=None)


@dataclass
class KsanaNodeGeneratorOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)


@dataclass
class KsanaNodeVAEEncodeOutput:
    samples: torch.Tensor = field(default=None)
    with_end_image: bool = field(default=False)
    batch_size_per_prompts: int = field(default=1)
