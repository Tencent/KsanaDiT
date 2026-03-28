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

"""Wan 2.2 I2V 14B — PipelineDef 声明（DAG 模式）。

导入此模块即自动注册到全局 PipelineDef 注册表。
I2V 流程：TextEncode → ReadImage(start/end) → VAE_ENCODE_SPATIAL → VACEPreprocess → Generate → VAE_DECODE → Save。
"""

from kdit.models.model_key import ModelKey
from kdit.nodes.core.node_types import InferNodeType as NT
from kdit.nodes.core.node_types import IONodeType as IOT
from kdit.tensor import TensorKey

from ..context_builders.wan import WanI2VContextBuilder
from ..pipeline_def import PipelineDefBuilder, register_pipeline_def
from ..pipeline_key import PipelineKey

_b = PipelineDefBuilder(PipelineKey.Wan2_2_I2V_14B)

# loaders
t5 = _b.add_loader(ModelKey.T5TextEncoder)
dit = _b.add_loader(ModelKey.Wan2_2_I2V_14B)
vae = _b.add_loader(ModelKey.VAE_WAN2_1)

# infer nodes
enc = _b.add_infer(NT.TEXT_ENCODE, ModelKey.T5TextEncoder)
read_start = _b.add_io(IOT.READ_IMAGE)
read_end = _b.add_io(IOT.READ_IMAGE)
vae_enc = _b.add_infer(NT.VAE_ENCODE_SPATIAL, ModelKey.VAE_WAN2_1)
vace = _b.add_infer(NT.VACE_PREPROCESS).when("has_vace")
gen = _b.add_infer(NT.GENERATE, ModelKey.Wan2_2_I2V_14B)
dec = _b.add_infer(NT.VAE_DECODE, ModelKey.VAE_WAN2_1)
save = _b.add_io(IOT.SAVE_VIDEO)

# edges
_b.connect(
    t5.T5TextEncoder >> enc.T5TextEncoder,
    vae.VAE_WAN2_1 >> vae_enc.VAE_WAN2_1,
    vae.VAE_WAN2_1 >> dec.VAE_WAN2_1,
    dit.Wan2_2_I2V_14B >> gen.Wan2_2_I2V_14B,
    # ReadImage → VAE_ENCODE_SPATIAL: 跨 pin 映射
    read_start.IMAGE >> vae_enc.START_IMG,
    read_end.IMAGE >> vae_enc.END_IMG,
    # VAE_ENCODE_SPATIAL → Generator
    vae_enc.BASE_LATENT >> gen.BASE_LATENT,
    # VACE → Generator
    vace.VACE_CONTEXT >> gen.VACE_CONTEXT,
    # text → Generator
    enc.POSITIVE >> gen.POSITIVE,
    enc.NEGATIVE >> gen.NEGATIVE,
    # Generator → Decode → Save
    gen.LATENTS >> dec.LATENTS,
    dec.VIDEO >> save.VIDEO,
)

WAN_I2V_14B_DEF = register_pipeline_def(_b.keep_tensors(TensorKey.VIDEO).context_builder(WanI2VContextBuilder).build())
