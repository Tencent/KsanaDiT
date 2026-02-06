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

from comfy.comfy_types.node_typing import IO

from ksana.nodes import KSANA_CATEGORY_UTILS
from ksana.utils import print_recursive


class KsanaDebugNode:

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        return {
            "required": {"source": (IO.ANY, {})},
            "optional": {"name": ("STRING", {"default": ""})},
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls):  # pylint: disable=invalid-name
        return True

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_UTILS

    def func(self, source, name="", node_id=None):
        name = name if name else "source"
        print(f"node_id {node_id} print {name}=")
        try:
            print_recursive(source)
        except Exception as e:  # pylint: disable=broad-except
            print(f"处理提示词时出错: {str(e)}")
        return ()
