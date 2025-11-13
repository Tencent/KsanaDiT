# WEB_DIRECTORY = "./web"
try:
    from .debug import kDitDebugNode
    from .cache import (
        kDitDCacheNode,
        kDitTeaCacheNode,
        kDitEasyCacheNode,
        kDitMagCacheNode,
    )
    from .model_loader import kDitModelLoaderNode
    from .generator import kDitGeneratorNode
except Exception as e:
    print(f"import错误: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "kDitDCacheNode": kDitDCacheNode,
    "kDitTeaCacheNode": kDitTeaCacheNode,
    "kDitEasyCacheNode": kDitEasyCacheNode,
    "kDitMagCacheNode": kDitMagCacheNode,
    "kDitDebugNode": kDitDebugNode,
    "kDitModelLoaderNode": kDitModelLoaderNode,
    "kDitGeneratorNode": kDitGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "kDitDCacheNode": "kDit DCache Node",
    "kDitTeaCacheNode": "kDit TeaCache Node",
    "kDitEasyCacheNode": "kDit EasyCache Node",
    "kDitMagCacheNode": "kDit MagCache Node",
    "kDitDebugNode": "kDit Debug Node",
    "kDitModelLoaderNode": "kDit Model Loader",
    "kDitGeneratorNode": "kDit Generator",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
