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
from .empty_latent import EmptyLatentNode
from .empty_torch_cache import KsanaEmptyTorchCacheNode
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
    "EmptyLatentNode": EmptyLatentNode,
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
    "KsanaEmptyTorchCacheNode": KsanaEmptyTorchCacheNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentNode": "kDiT EmptyLatent",
    "KsanaAttentionConfigNode": "kDiT AttentionConfig",
    "KsanaCustomStepCacheNode": "kDiT CustomStepCache",
    "KsanaHybridCacheNode": "kDiT HybridCache",
    "KsanaCacheCombineNode": "kDiT CacheCombine",
    "KsanaDCacheNode": "kDiT DCache",
    "KsanaTeaCacheNode": "kDiT TeaCache",
    "KsanaEasyCacheNode": "kDiT EasyCache",
    "KsanaMagCacheNode": "kDiT MagCache",
    "KsanaDBCacheNode": "kDiT DBCache",
    "KsanaDebugNode": "kDiT DebugNode",
    "KsanaModelLoaderNode": "kDiT Model Loader",
    "KsanaVAELoaderNode": "kDiT VAE Loader",
    "KsanaVAEEncodeNode": "kDiT VAE Encoder",
    "KsanaVAEDecodeNode": "kDiT VAE Decoder",
    "KsanaGeneratorNode": "kDiT Generator",
    "KsanaTorchCompileNode": "kDiT TorchCompile",
    "KsanaLoraSelectMultiNode": "kDiT LoraSelectMulti",
    "KsanaLoraSelectNode": "kDiT LoraSelect",
    "KsanaLoraCombineNode": "kDiT LoraCombine",
    "KsanaRadialSageAttentionConfigNode": "kDiT RadialSageAttentionConfig",
    "KsanaVAEImageEncodeNode": "kDiT VAE Image Encoder ",
    "KsanaVaceModelSelectNode": "kDiT Vace Model Select",
    "KsanaVideoControlConfigNode": "kDiT VideoControlConfig",
    "KsanaWanVaceToVideoNode": "kDiT WanVace To Video",
    "KsanaSLGNode": "kDiT Skip Layer Guidance",
    "KsanaEnhanceAVideoNode": "kDiT Enhance-A-Video",
    "KsanaExperimentalArgsNode": "kDiT Experimental Args",
    "KsanaSageSLAConfigNode": "kDiT SageSLAttentionConfig",
    "KsanaTextEmbConverterNode": "kDiT TextEmbConverter",
    "KsanaEmptyTorchCacheNode": "kDiT Empty Torch Cache",
}

# WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
