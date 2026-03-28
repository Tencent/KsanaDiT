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

import os

from kdit import get_engine
from kdit.config import DistributedConfig, KsanaAttentionConfig, KsanaLinearBackend, ModelConfig
from kdit.models.model_key import get_model_key_from_path
from kdit.nodes.core.node_context import NodeContext
from kdit.nodes.core.node_def import NodeDef
from kdit.nodes.core.node_types import IONodeType
from kdit.utils import get_gpu_count, log
from kdit.utils.profile import MemoryProfiler

from .output_types import ModelLoaderOutput


class KsanaNodeModelLoader:
    LOADED_MODEL = None

    @classmethod
    def load(
        cls,
        high_noise_model_path: str,
        low_noise_model_path: str = None,
        vace_model: list[str] | None = None,
        run_dtype="float16",
        rms_dtype="float",
        linear_backend: KsanaLinearBackend | str = KsanaLinearBackend.DEFAULT,
        attention_config: KsanaAttentionConfig | None = None,
        model_boundary=None,
        torch_compile_args=None,
        lora=None,
        comfy_progress_bar_func=None,
    ):
        # Qwen-Image is much more stable in bfloat16; fp16 frequently overflows to NaN in practice.
        # Our own example script uses bfloat16 by default.
        if "qwen" in high_noise_model_path.lower() and "image" in high_noise_model_path.lower():
            if run_dtype in ("float16", "fp16", "torch.float16"):
                log.warning("qwen-image detected: forcing run_dtype to bfloat16 for numerical stability.")
                run_dtype = "bfloat16"

        num_gpus = get_gpu_count()
        if comfy_progress_bar_func is None:
            comfyui_progress_bar = None
        else:
            comfyui_progress_bar = comfy_progress_bar_func(1 if low_noise_model_path is None else 2)

        def comfy_bar_callback():
            if comfyui_progress_bar is None:
                return
            comfyui_progress_bar.update(1)

        model_config = ModelConfig(
            run_dtype=run_dtype,
            rms_dtype=rms_dtype,
            linear_backend=KsanaLinearBackend(linear_backend),
            attention_config=KsanaAttentionConfig() if attention_config is None else attention_config,
            torch_compile_config=torch_compile_args,
            boundary=model_boundary,
        )

        MemoryProfiler.record_memory("before_load_model")
        if not high_noise_model_path:
            raise ValueError("high_noise_model_path is empty; check ComfyUI diffusion_models paths.")
        if not os.path.exists(high_noise_model_path):
            raise FileNotFoundError(f"high_noise_model_path not found: {high_noise_model_path}")
        if low_noise_model_path is not None and not os.path.exists(low_noise_model_path):
            raise FileNotFoundError(f"low_noise_model_path not found: {low_noise_model_path}")

        model_path = (
            (high_noise_model_path, low_noise_model_path) if low_noise_model_path is not None else high_noise_model_path
        )
        model_key = get_model_key_from_path(model_path)

        kdit_engine = get_engine(dist_config=DistributedConfig(num_gpus=num_gpus))
        if cls.LOADED_MODEL is not None:
            kdit_engine.clear_models(cls.LOADED_MODEL.pin)

        try:
            node_def = NodeDef(node_type=IONodeType.LOAD_MODEL, model_key=model_key)
            context = NodeContext(
                metadata={
                    "model_path": model_path,
                    "model_patch_path": vace_model,
                    "model_config": model_config,
                    "lora_config": lora,
                    "comfy_bar_callback": comfy_bar_callback,
                }
            )
            output_pins = kdit_engine.run_node(node_def, {}, context)
            model_pool_key = output_pins.get(model_key)
        except Exception as e:  # pylint: disable=broad-except
            cls.LOADED_MODEL = None
            log.exception("load_diffusion_model failed")
            raise RuntimeError(
                f"load_diffusion_model failed for: {high_noise_model_path} ({type(e).__name__}: {e})"
            ) from e

        cls.LOADED_MODEL = model_pool_key
        MemoryProfiler.record_memory("after_load_model")
        return ModelLoaderOutput(
            model=model_pool_key,
            run_dtype=model_config.run_dtype,
        )
