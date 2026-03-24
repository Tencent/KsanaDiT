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

"""Qwen Image Edit — PipelineDef 声明（DAG 模式）。

导入此模块即自动注册到全局 PipelineDef 注册表。
Edit 流程：TextEncode → ReadImage → VAE_ENCODE_IMAGES(条件)
→ VAE_COMPUTE_SHAPE → Generate → VAE_DECODE → SaveImage。
"""

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.tensor import TensorKey

from ..context_builders.qwen import QwenEditContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

_b = PipelineDefBuilder(PipelineKey.QwenImage_Edit)

# loaders
qwen = _b.add_loader(ModelKey.Qwen2VLTextEncoderMultimodal)
dit = _b.add_loader(ModelKey.QwenImage_Edit)
vae = _b.add_loader(ModelKey.QwenImageVAE)

# infer nodes
text = _b.add_infer(NT.TEXT_ENCODE, ModelKey.Qwen2VLTextEncoderMultimodal)
read_r = _b.add_infer(NT.READ_IMAGE)
vae_enc = _b.add_infer(NT.VAE_ENCODE_IMAGES, ModelKey.QwenImageVAE).when("has_ref_images")
shape = _b.add_infer(NT.VAE_COMPUTE_SHAPE, ModelKey.QwenImageVAE)
gen = _b.add_infer(NT.GENERATE, ModelKey.QwenImage_Edit)
dec = _b.add_infer(NT.VAE_DECODE, ModelKey.QwenImageVAE)
save = _b.add_infer(NT.SAVE_IMAGE)

# edges
_b.connect(
    qwen.Qwen2VLTextEncoderMultimodal >> text.Qwen2VLTextEncoderMultimodal,
    vae.QwenImageVAE >> vae_enc.QwenImageVAE,
    vae.QwenImageVAE >> shape.QwenImageVAE,
    vae.QwenImageVAE >> dec.QwenImageVAE,
    dit.QwenImage_Edit >> gen.QwenImage_Edit,
    read_r.IMAGE >> vae_enc.IMAGE,
    vae_enc.AUX_LATENT >> gen.AUX_LATENT,
    text.POSITIVE >> gen.POSITIVE,
    text.NEGATIVE >> gen.NEGATIVE,
    shape.BASE_LATENT >> gen.BASE_LATENT,
    gen.LATENTS >> dec.LATENTS,
    dec.VIDEO >> save.VIDEO,
)

QWEN_EDIT_DEF = register_pipeline_def(_b.keep_tensors(TensorKey.VIDEO).context_builder(QwenEditContextBuilder).build())
