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

"""Wan 2.1 VACE 14B — PipelineDef 声明。

导入此模块即自动注册到全局 PipelineDef 注册表。
VACE 与 I2V 共享 WanI2VContextBuilder — 两者的区别仅在 PipelineDef 的模型配置。
"""

from kdit.models.model_key import KsanaModelKey
from kdit.nodes.core.node_types import KsanaInferNodeType
from kdit.tensor import TensorKey

from ..context_builders.wan import WanI2VContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

WAN_VACE_14B_DEF = register_pipeline_def(
    PipelineDefBuilder(PipelineKey.Wan2_1_VACE_14B)
    .load("text_encoder", KsanaModelKey.T5TextEncoder)
    .load("diffusion", KsanaModelKey.Wan2_1_VACE_14B)
    .load("vae", KsanaModelKey.VAE_WAN2_1)
    .add_infer(KsanaInferNodeType.TEXT_ENCODE, model_role="text_encoder")
    .add_infer(KsanaInferNodeType.VAE_ENCODE_SPATIAL, model_role="vae")
    .when("has_start_image")
    .add_infer(KsanaInferNodeType.GENERATE, model_role="diffusion")
    .add_infer(KsanaInferNodeType.VAE_DECODE, model_role="vae")
    .add_infer(KsanaInferNodeType.SAVE_VIDEO)
    .keep_tensors(TensorKey.VIDEO)
    .context_builder(WanI2VContextBuilder)
    .build()
)
