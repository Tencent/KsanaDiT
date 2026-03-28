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

"""Qwen Image T2I — PipelineDef 声明（DAG 模式）。

导入此模块即自动注册到全局 PipelineDef 注册表。
"""

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.nodes.core.node_types import IONodeType as IOT
from kdit.tensor import TensorKey

from ..context_builders.qwen import QwenT2IContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

_b = PipelineDefBuilder(PipelineKey.QwenImage_T2I)

# loaders
qwen = _b.add_loader(ModelKey.Qwen2VLTextEncoder)
dit = _b.add_loader(ModelKey.QwenImage_T2I)
vae = _b.add_loader(ModelKey.QwenImageVAE)

# infer nodes
text = _b.add_infer(NT.TEXT_ENCODE, ModelKey.Qwen2VLTextEncoder)
shape = _b.add_infer(NT.VAE_COMPUTE_SHAPE, ModelKey.QwenImageVAE)
gen = _b.add_infer(NT.GENERATE, ModelKey.QwenImage_T2I)
dec = _b.add_infer(NT.VAE_DECODE, ModelKey.QwenImageVAE)
save = _b.add_io(IOT.SAVE_IMAGE)

# edges
_b.connect(
    qwen.Qwen2VLTextEncoder >> text.Qwen2VLTextEncoder,
    vae.QwenImageVAE >> shape.QwenImageVAE,
    vae.QwenImageVAE >> dec.QwenImageVAE,
    dit.QwenImage_T2I >> gen.QwenImage_T2I,
    text.POSITIVE >> gen.POSITIVE,
    text.NEGATIVE >> gen.NEGATIVE,
    shape.BASE_LATENT >> gen.BASE_LATENT,
    gen.LATENTS >> dec.LATENTS,
    dec.VIDEO >> save.VIDEO,
)

QWEN_T2I_DEF = register_pipeline_def(_b.keep_tensors(TensorKey.VIDEO).context_builder(QwenT2IContextBuilder).build())
