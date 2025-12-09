# WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .debug import KsanaDebugNode
    from .cache import (
        KsanaDCacheNode,
        KsanaTeaCacheNode,
        KsanaEasyCacheNode,
        KsanaMagCacheNode,
    )
    from .model_loader import KsanaModelLoaderNode
    from .generator import KsanaGeneratorNode
    from .torch_compile import KsanaTorchCompileNode
    from .lora import KsanaLoraSelectMultiNode, KsanaLoraSelectNode, KsanaLoraCombineNode

    NODE_CLASS_MAPPINGS = {
        "KsanaDCacheNode": KsanaDCacheNode,
        "KsanaTeaCacheNode": KsanaTeaCacheNode,
        "KsanaEasyCacheNode": KsanaEasyCacheNode,
        "KsanaMagCacheNode": KsanaMagCacheNode,
        "KsanaDebugNode": KsanaDebugNode,
        "KsanaModelLoaderNode": KsanaModelLoaderNode,
        "KsanaGeneratorNode": KsanaGeneratorNode,
        "KsanaTorchCompileNode": KsanaTorchCompileNode,
        "KsanaLoraSelectMultiNode": KsanaLoraSelectMultiNode,
        "KsanaLoraSelectNode": KsanaLoraSelectNode,
        "KsanaLoraCombineNode": KsanaLoraCombineNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "KsanaDCacheNode": "KsanaDit DCache Node",
        "KsanaTeaCacheNode": "KsanaDit TeaCache Node",
        "KsanaEasyCacheNode": "KsanaDit EasyCache Node",
        "KsanaMagCacheNode": "KsanaDit MagCache Node",
        "KsanaDebugNode": "KsanaDit Debug Node",
        "KsanaModelLoaderNode": "KsanaDit Model Loader",
        "KsanaGeneratorNode": "KsanaDit Generator",
        "KsanaTorchCompileNode": "KsanaDiT Torch Compile Node",
        "KsanaLoraSelectMultiNode": "KsanaDit Lora Select Multi Node",
        "KsanaLoraSelectNode": "KsanaDit Lora Select Node",
        "KsanaLoraCombineNode": "KsanaDit Lora Combine Node",
    }
except Exception as e:
    print(f"import错误: {str(e)}")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
