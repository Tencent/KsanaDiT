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

from ksana.config import KsanaTorchCompileConfig


def torch_compile_config(
    backend,
    fullgraph,
    mode,
    dynamic,
    compile_transformer_blocks_only,
    dynamo_cache_size_limit=128,
    dynamo_recompile_limit=128,
    force_parameter_static_shapes=False,
    allow_unmerged_lora_compile=False,
):
    return KsanaTorchCompileConfig(
        backend=backend,
        fullgraph=fullgraph,
        mode=mode,
        dynamic=dynamic,
        compile_transformer_blocks_only=compile_transformer_blocks_only,
        dynamo_cache_size_limit=dynamo_cache_size_limit,
        dynamo_recompile_limit=dynamo_recompile_limit,
        force_parameter_static_shapes=force_parameter_static_shapes,
        allow_unmerged_lora_compile=allow_unmerged_lora_compile,
    )
