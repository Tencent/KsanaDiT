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

import ksana.nodes as nodes
from ksana.nodes import (
    KSANA_CATEGORY_CONVERTER,
    KSANA_TEXT_ENCODE_OUTPUT,
)


class KsanaTextEmbConverterNode:
    @classmethod
    def INPUT_TYPES(s):  # pylint: disable=invalid-name
        return {
            "required": {
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
            },
        }

    RETURN_TYPES = (KSANA_TEXT_ENCODE_OUTPUT, KSANA_TEXT_ENCODE_OUTPUT)
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "func"
    CATEGORY = KSANA_CATEGORY_CONVERTER
    DESCRIPTION = "Convert text embeds to KSANA_TEXT_ENCODE_OUTPUT format."

    def func(self, text_embeds):
        return nodes.convert_text_embeds_to_ksana(text_embeds)
