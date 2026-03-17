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

"""Qwen Image T2I — PipelineDef 声明。

导入此模块即自动注册到全局 PipelineDef 注册表。
"""

from kdit.models.model_key import KsanaModelKey
from kdit.nodes.core.node_types import KsanaInferNodeType
from kdit.tensor import TensorKey

from ..context_builders.qwen import QwenT2IContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

QWEN_T2I_DEF = register_pipeline_def(
    PipelineDefBuilder(PipelineKey.QwenImage_T2I)
    .load("text_encoder", KsanaModelKey.Qwen2VLTextEncoder)
    .load("diffusion", KsanaModelKey.QwenImage_T2I)
    .load("vae", KsanaModelKey.QwenImageVAE)
    .add_infer(KsanaInferNodeType.TEXT_ENCODE, model_role="text_encoder")
    .add_infer(KsanaInferNodeType.GENERATE, model_role="diffusion")
    .add_infer(KsanaInferNodeType.VAE_DECODE, model_role="vae")
    .add_infer(KsanaInferNodeType.SAVE_IMAGE)
    .keep_tensors(TensorKey.VIDEO)
    .context_builder(QwenT2IContextBuilder)
    .build()
)
