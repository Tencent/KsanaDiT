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

"""Qwen Image Edit — PipelineDef 声明。

导入此模块即自动注册到全局 PipelineDef 注册表。
Edit 流程：TextEncode → (VAE_ENCODE_IMAGES when has_ref_images) → Generate → VAE_DECODE → SaveImage。
"""

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType
from kdit.tensor import TensorKey

from ..context_builders.qwen import QwenEditContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

QWEN_EDIT_DEF = register_pipeline_def(
    PipelineDefBuilder(PipelineKey.QwenImage_Edit)
    .load(ModelKey.Qwen2VLTextEncoderMultimodal)
    .load(ModelKey.QwenImage_Edit)
    .load(ModelKey.QwenImageVAE)
    .add_infer(InferNodeType.TEXT_ENCODE, ModelKey.Qwen2VLTextEncoderMultimodal)
    .add_infer(InferNodeType.VAE_COMPUTE_SHAPE, ModelKey.QwenImageVAE)
    .add_infer(InferNodeType.VAE_ENCODE_IMAGES, ModelKey.QwenImageVAE)
    .when("has_ref_images")
    .add_infer(InferNodeType.GENERATE, ModelKey.QwenImage_Edit)
    .add_infer(InferNodeType.VAE_DECODE, ModelKey.QwenImageVAE)
    .add_infer(InferNodeType.SAVE_IMAGE)
    .keep_tensors(TensorKey.VIDEO)
    .context_builder(QwenEditContextBuilder)
    .build()
)
