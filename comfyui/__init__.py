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
    "kDitDCacheNode": "KsanaDit DCache Node",
    "kDitTeaCacheNode": "KsanaDit TeaCache Node",
    "kDitEasyCacheNode": "KsanaDit EasyCache Node",
    "kDitMagCacheNode": "KsanaDit MagCache Node",
    "kDitDebugNode": "KsanaDit Debug Node",
    "kDitModelLoaderNode": "KsanaDit Model Loader",
    "kDitGeneratorNode": "KsanaDit Generator",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
