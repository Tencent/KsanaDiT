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

import torch

from ..accelerator import platform
from . import log

try:
    # Avoid Dynamo compiling cache helpers that use numpy/Python control flow
    from torch._dynamo import disable as disable_dynamo  # noqa: F401  # pylint: disable=unused-import
except ImportError:

    def disable_dynamo(fn=None):
        return fn if fn is not None else (lambda f: f)


def apply_torch_compile(model, torch_compile_config=None):
    if platform.is_npu() or torch_compile_config is None:  # TODO: support torch compile in NPU
        return model
    log.info(f"apply torch_compile_config: {torch_compile_config}")
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = torch_compile_config.dynamo_cache_size_limit
        torch._dynamo.config.force_parameter_static_shapes = torch_compile_config.force_parameter_static_shapes
        try:
            torch._dynamo.config.recompile_limit = torch_compile_config.dynamo_recompile_limit
        except Exception as e:  # pylint: disable=broad-except
            log.warning(f"Could not set recompile_limit: {e}")

    if torch_compile_config.compile_transformer_blocks_only:
        log.info("Compiling only transformer blocks")

        block_attrs = getattr(model, "_compilable_block_attrs", None)
        if block_attrs is None:
            block_attrs = ["blocks", "transformer_blocks"]
            log.info("Model does not declare _compilable_block_attrs, " f"falling back to default: {block_attrs}")

        compiled_total = 0
        found_any = False
        for attr_name in block_attrs:
            blocks = getattr(model, attr_name, None)
            if blocks is None:
                continue
            found_any = True
            compiled_cnt = 0
            for i, block in enumerate(blocks):
                try:
                    blocks[i] = torch.compile(
                        block,
                        backend=torch_compile_config.backend,
                        mode=torch_compile_config.mode,
                        fullgraph=torch_compile_config.fullgraph,
                        dynamic=torch_compile_config.dynamic,
                    )
                    compiled_cnt += 1
                except Exception as e:  # pylint: disable=broad-except
                    log.warning(f"torch.compile {attr_name}[{i}] failed: {e}")
            log.info(f"Applied torch.compile to {compiled_cnt}/{len(blocks)} blocks in '{attr_name}'.")
            compiled_total += compiled_cnt

        if not found_any:
            log.warning(f"No compilable blocks found (checked attributes: {block_attrs})")
            return model

        log.info(f"Total compiled blocks: {compiled_total}.")
    else:
        log.info("Compiling entire model")
        model = torch.compile(
            model,
            fullgraph=torch_compile_config.fullgraph,
            dynamic=torch_compile_config.dynamic,
            backend=torch_compile_config.backend,
            mode=torch_compile_config.mode,
        )
    return model
