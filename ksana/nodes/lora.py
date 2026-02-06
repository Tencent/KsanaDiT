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

from ksana.config.lora_config import KsanaLoraConfig


def build_list_of_lora_config(lora_inputs: list[tuple[str, float]] | tuple[str, float]):
    loras_list = []
    if not isinstance(lora_inputs, list):
        lora_inputs = [lora_inputs]
    for lora_path, strength in lora_inputs:
        if isinstance(strength, list):
            raise ValueError(f"lora strength must be a scalar, but got {strength}")
        if lora_path is None:
            continue
        loras_list.append(KsanaLoraConfig(path=lora_path, strength=round(strength, 4)))
    return loras_list
