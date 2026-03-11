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

KSANA_DIFFUSION_MODEL = "KSANA_DIFFUSION_MODEL"
KSANA_LORA = "KSANA_LORA"
KSANA_VAE_MODEL = "KSANA_VAE_MODEL"
KSANA_VACE_MODEL = "KSANA_VACE_MODEL"

KSANA_VAE_ENCODE_OUTPUT = "KSANA_VAE_ENCODE_OUTPUT"
KSANA_GENERATE_OUTPUT = "KSANA_GENERATE_OUTPUT"
KSANA_TEXT_ENCODE_OUTPUT = "CONDITIONING"

KSANA_ATTENTION_CONFIG = "KSANA_ATTENTION_CONFIG"
KSANA_CACHE_CONFIG = "KSANA_CACHE_CONFIG"
KSANA_TORCH_COMPILE = "KSANA_TORCH_COMPILE"
KSANA_VIDEO_CONTROL_CONFIG = "KSANA_VIDEO_CONTROL_CONFIG"

KSANA_SLG_ARGS = "KSANA_SLG_ARGS"
KSANA_FETA_ARGS = "KSANA_FETA_ARGS"
KSANA_EXPERIMENTAL_ARGS = "KSANA_EXPERIMENTAL_ARGS"
KSANA_VACE_EMBEDS = "KSANA_VACE_EMBEDS"

# WanVideoWrapper compatible types (for cross-project compatibility)
WANVIDEO_SLG_ARGS = "SLGARGS"
WANVIDEO_FETA_ARGS = "FETAARGS"
WANVIDEO_EXPERIMENTAL_ARGS = "EXPERIMENTALARGS"

KSANA_ANY_TYPE = type("_AnyType", (str,), {"__ne__": lambda self, _: False})("*")

KSANA_CATEGORY_LORA = "ksana/lora"
KSANA_CATEGORY_CACHE = "ksana/cache"
KSANA_CATEGORY_VAE = "ksana/vae"
KSANA_CATEGORY_UTILS = "ksana/utils"
KSANA_CATEGORY_CONFIGS = "ksana/configs"
KSANA_CATEGORY_CONVERTER = "ksana/converter"
