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

import ksana.nodes as nodes
from ksana.nodes.types import KSANA_ANY_TYPE


class KsanaEmptyTorchCacheNode:
    """Passthrough node that frees GPU/NPU memory."""

    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "any": (KSANA_ANY_TYPE, {"tooltip": "Any input; will be passed through unchanged."}),
            },
        }

    RETURN_TYPES = (KSANA_ANY_TYPE,)
    RETURN_NAMES = ("any",)
    FUNCTION = "run"
    CATEGORY = "ksana"
    DESCRIPTION = (
        "Empties the GPU/NPU memory cache. "
        "Insert after KsanaDiT nodes and before memory-intensive post-processing nodes."
    )

    def run(self, any):  # pylint: disable=redefined-builtin
        nodes.ksana_empty_torch_cache()
        return (any,)
