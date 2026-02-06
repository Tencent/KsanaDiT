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

from .attn_config import KsanaAttentionConfigNode, KsanaRadialSageAttentionConfigNode, KsanaSageSLAConfigNode
from .cache import (
    KsanaCacheCombineNode,
    KsanaCustomStepCacheNode,
    KsanaDBCacheNode,
    KsanaDCacheNode,
    KsanaEasyCacheNode,
    KsanaHybridCacheNode,
    KsanaMagCacheNode,
    KsanaTeaCacheNode,
)
from .converter import KsanaTextEmbConverterNode
from .debug import KsanaDebugNode
from .generator import KsanaGeneratorNode
from .lora import KsanaLoraCombineNode, KsanaLoraSelectMultiNode, KsanaLoraSelectNode
from .model_loader import KsanaModelLoaderNode, KsanaVaceModelSelectNode
from .torch_compile import KsanaTorchCompileNode
from .vace import (
    KsanaEnhanceAVideoNode,
    KsanaExperimentalArgsNode,
    KsanaSLGNode,
    KsanaVideoControlConfigNode,
    KsanaWanVaceToVideoNode,
)
from .vae import KsanaVAEDecodeNode, KsanaVAEEncodeNode, KsanaVAEImageEncodeNode, KsanaVAELoaderNode

NODE_CLASS_MAPPINGS = {
    "KsanaAttentionConfigNode": KsanaAttentionConfigNode,
    "KsanaCustomStepCacheNode": KsanaCustomStepCacheNode,
    "KsanaHybridCacheNode": KsanaHybridCacheNode,
    "KsanaCacheCombineNode": KsanaCacheCombineNode,
    "KsanaDCacheNode": KsanaDCacheNode,
    "KsanaTeaCacheNode": KsanaTeaCacheNode,
    "KsanaEasyCacheNode": KsanaEasyCacheNode,
    "KsanaMagCacheNode": KsanaMagCacheNode,
    "KsanaDBCacheNode": KsanaDBCacheNode,
    "KsanaDebugNode": KsanaDebugNode,
    "KsanaModelLoaderNode": KsanaModelLoaderNode,
    "KsanaVAELoaderNode": KsanaVAELoaderNode,
    "KsanaVAEEncodeNode": KsanaVAEEncodeNode,
    "KsanaVAEDecodeNode": KsanaVAEDecodeNode,
    "KsanaGeneratorNode": KsanaGeneratorNode,
    "KsanaTorchCompileNode": KsanaTorchCompileNode,
    "KsanaLoraSelectMultiNode": KsanaLoraSelectMultiNode,
    "KsanaLoraSelectNode": KsanaLoraSelectNode,
    "KsanaLoraCombineNode": KsanaLoraCombineNode,
    "KsanaRadialSageAttentionConfigNode": KsanaRadialSageAttentionConfigNode,
    "KsanaVAEImageEncodeNode": KsanaVAEImageEncodeNode,
    "KsanaVaceModelSelectNode": KsanaVaceModelSelectNode,
    "KsanaVideoControlConfigNode": KsanaVideoControlConfigNode,
    "KsanaWanVaceToVideoNode": KsanaWanVaceToVideoNode,
    "KsanaSLGNode": KsanaSLGNode,
    "KsanaEnhanceAVideoNode": KsanaEnhanceAVideoNode,
    "KsanaExperimentalArgsNode": KsanaExperimentalArgsNode,
    "KsanaSageSLAConfigNode": KsanaSageSLAConfigNode,
    "KsanaTextEmbConverterNode": KsanaTextEmbConverterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KsanaAttentionConfigNode": "KsanaDit AttentionConfig",
    "KsanaCustomStepCacheNode": "KsanaDit CustomStepCache",
    "KsanaHybridCacheNode": "KsanaDit HybridCache",
    "KsanaCacheCombineNode": "KsanaDit CacheCombine",
    "KsanaDCacheNode": "KsanaDit DCache",
    "KsanaTeaCacheNode": "KsanaDit TeaCache",
    "KsanaEasyCacheNode": "KsanaDit EasyCache",
    "KsanaMagCacheNode": "KsanaDit MagCache",
    "KsanaDBCacheNode": "KsanaDit DBCache",
    "KsanaDebugNode": "KsanaDit DebugNode",
    "KsanaModelLoaderNode": "KsanaDit Model Loader",
    "KsanaVAELoaderNode": "KsanaDit VAE Loader",
    "KsanaVAEEncodeNode": "KsanaDit VAE Encoder",
    "KsanaVAEDecodeNode": "KsanaDit VAE Decoder",
    "KsanaGeneratorNode": "KsanaDit Generator",
    "KsanaTorchCompileNode": "KsanaDiT TorchCompile",
    "KsanaLoraSelectMultiNode": "KsanaDit LoraSelectMulti",
    "KsanaLoraSelectNode": "KsanaDit LoraSelect",
    "KsanaLoraCombineNode": "KsanaDit LoraCombine",
    "KsanaRadialSageAttentionConfigNode": "KsanaDit RadialSageAttentionConfig",
    "KsanaVAEImageEncodeNode": "KsanaDit VAE Image Encoder ",
    "KsanaVaceModelSelectNode": "KsanaDit Vace Model Select",
    "KsanaVideoControlConfigNode": "KsanaDit VideoControlConfig",
    "KsanaWanVaceToVideoNode": "KsanaDit WanVace To Video",
    "KsanaSLGNode": "KsanaDit Skip Layer Guidance",
    "KsanaEnhanceAVideoNode": "KsanaDit Enhance-A-Video",
    "KsanaExperimentalArgsNode": "KsanaDit Experimental Args",
    "KsanaSageSLAConfigNode": "KsanaDit SageSLAttentionConfig",
    "KsanaTextEmbConverterNode": "KsanaDit TextEmbConverter",
}

# WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
