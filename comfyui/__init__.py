# WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .debug import KsanaDebugNode
    from .cache import (
        KsanaDCacheNode,
        KsanaCustomStepCacheNode,
        KsanaHybridCacheNode,
        KsanaCacheCombineNode,
        KsanaTeaCacheNode,
        KsanaEasyCacheNode,
        KsanaMagCacheNode,
        KsanaDBCacheNode,
    )
    from .model_loader import KsanaModelLoaderNode
    from .vae import KsanaVAELoaderNode, KsanaVAEEncodeNode, KsanaVAEDecodeNode
    from .generator import KsanaGeneratorNode
    from .torch_compile import KsanaTorchCompileNode
    from .lora import KsanaLoraSelectMultiNode, KsanaLoraSelectNode, KsanaLoraCombineNode

    NODE_CLASS_MAPPINGS = {
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
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
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
    }
except Exception as e:
    print(f"import错误: {str(e)}")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
